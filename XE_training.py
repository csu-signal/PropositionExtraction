import pickle
import sys
import torch
from helper import tokenize, forward_ab, f1_score, accuracy, precision, recall
import pandas as pd
import random
from tqdm import tqdm
import os
from models import CrossEncoder
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
import re
from transformers import AutoTokenizer
import torch
from cosine_sim import CrossEncoder_cossim
import torch.nn.functional as F
from helperMethods import is_proposition_present, normalize_expression, normalize_sub_expression, extract_colors, extract_colors_and_numbers
##These methods are used for pruning 
# Mapping words to numbers for comparison


def broaden_search_with_numbers(common_grounds, mentioned_numbers):
    # Filter common grounds to include those mentioning any of the mentioned numbers
    
    broadened_common_grounds = [cg for cg in common_grounds if any(str(number) in cg for number in mentioned_numbers)]
    
    return broadened_common_grounds
def broaden_search_with_colors(common_grounds, mentioned_colors):
    # Filter common grounds to include those mentioning any of the mentioned colors
    broadened_common_grounds = [cg for cg in common_grounds if any(color in cg for color in mentioned_colors)]
    return broadened_common_grounds

#used for when the numbers are spelled out 
number_mapping = {
    "ten": 10, "twenty": 20, "thirty": 30, 
    "forty": 40, "fifty": 50
}


def is_valid_common_ground(cg, elements):
    cg_colors = re.findall(r'\b(?:red|blue|green|yellow|purple)\b', cg)
    cg_numbers = [int(num) for num in re.findall(r'\b(?:10|20|30|40|50)\b', cg)]
    color_match = not elements["colors"] or set(cg_colors) == set(elements["colors"])
    number_match = not elements["numbers"] or set(cg_numbers) == set(elements["numbers"])
    return color_match and number_match

def is_valid_individual_match(cg, elements):
    cg_colors = re.findall(r'\b(?:red|blue|green|yellow|purple)\b', cg)
    cg_numbers = [int(num) for num in re.findall(r'\b(?:10|20|30|40|50)\b', cg)]
    for color in elements["colors"]:
        for number in elements["numbers"]:
            if color in cg_colors and number in cg_numbers:
                return True
    return False
    

def make_proposition_map(dataset):
    data = f'./Data/goldenFiles/{dataset}.csv'
    df = pd.read_csv(data)
    prop_dict = defaultdict(dict)
    for x, y in enumerate(df.iterrows()):

        prop_dict[x]['common_ground'] = df['Common Ground'][x]
        prop_dict[x]['transcript'] = df['Transcript'][x]
        prop_dict[x]['label'] = df['Label'][x]
        prop_dict[x]['group'] = df['Group'][x]
    return prop_dict, df



def add_special_tokens(proposition_map):
    for x, y in proposition_map.items():
        #print(y['common_ground'])
        cg_with_token = "<m>" + " " + y['common_ground']+ " "  + "</m>"
        #print(y['transcript'])
        prop_with_token = "<m>" + " "+ y['transcript'] +" " + "</m>"
        proposition_map[x]['common_ground'] = cg_with_token
        proposition_map[x]['transcript'] = prop_with_token
    return proposition_map

def predict_with_XE(parallel_model, dev_ab, dev_ba, device, batch_size, cosine_sim=False):
    n = dev_ab['input_ids'].shape[0]
    indices = list(range(n))
    # new_batch_size = batching(n, batch_size, len(device_ids))
    # batch_size = new_batch_size
    all_scores_ab = []
    all_scores_ba = []
    description='Predicting'
    if(cosine_sim):
        description = 'Getting Cosine'
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_indices = indices[i: i + batch_size]
            scores_ab = forward_ab(parallel_model, dev_ab, device, batch_indices,cosine_sim=False)
            scores_ba = forward_ab(parallel_model, dev_ba, device, batch_indices,cosine_sim=False)
            all_scores_ab.append(scores_ab.detach().cpu())
            all_scores_ba.append(scores_ba.detach().cpu())

    return torch.cat(all_scores_ab), torch.cat(all_scores_ba) 


def train_prop_XE(dataset, model_name=None,n_splits=10, resultsFolder = ''):
    dataset_folder = f'./datasets/{dataset}/'
    device = torch.device('cuda:1')
    #device_ids = list(range(1))
    device_ids = [1]
    #load the statement and proposition data
    prop_dict, df = make_proposition_map("oraclePreprocessedlevel1")
    proposition_map = add_special_tokens(prop_dict)
 
    groups = df['Group'].values

    # Setting up group k-fold cross-validation
    gkf = GroupKFold(n_splits=n_splits)
    
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=groups)):
        print(f"Training on fold {fold+1}")

        train_pairs = train_idx.tolist()  # Convert numpy array to list
        train_labels = [df['Label'].iloc[idx] for idx in train_idx]
        dev_pairs = test_idx.tolist()  # Convert numpy array to list
        dev_labels = [df['Label'].iloc[idx] for idx in test_idx]
        group = df['Group'].iloc[test_idx[0]]
       
        scorer_module = CrossEncoder(is_training=True,long=False,  model_name=model_name).to(device)

        parallel_model = torch.nn.DataParallel(scorer_module, device_ids=device_ids)
        parallel_model.module.to(device)
        
        train(train_pairs, train_labels, dev_pairs, dev_labels, parallel_model, proposition_map, dataset_folder, device,
            batch_size=20, n_iters=12, lr_lm=0.000001, lr_class=0.0001,group =group, resultsFolder = resultsFolder)
        
        
 
  
def tokenize_props(tokenizer, proposition_ids, proposition_map, m_end, max_sentence_len=1024, truncate=True):
    if max_sentence_len is None:
        max_sentence_len = tokenizer.model_max_length

    pairwise_bert_instances_ab = []
    pairwise_bert_instances_ba = []

    doc_start = '<doc-s>'
    doc_end = '</doc-s>'

    for index in proposition_ids:
        sentence_a = proposition_map[index]['transcript']
        sentence_b = proposition_map[index]['common_ground']

        def make_instance(sent_a, sent_b):
            return ' '.join(['<g>', doc_start, sent_a, doc_end]), \
                   ' '.join([doc_start, sent_b, doc_end])

        instance_ab = make_instance(sentence_a, sentence_b)
        pairwise_bert_instances_ab.append(instance_ab)

        instance_ba = make_instance(sentence_b, sentence_a)
        pairwise_bert_instances_ba.append(instance_ba)

    def truncate_with_mentions(input_ids):
        input_ids_truncated = []
        for input_id in input_ids:
            m_end_index = input_id.index(m_end)

            curr_start_index = max(0, m_end_index - (max_sentence_len // 4))

            in_truncated = input_id[curr_start_index: m_end_index] + \
                           input_id[m_end_index: m_end_index + (max_sentence_len // 4)]
            in_truncated = in_truncated + [tokenizer.pad_token_id] * (max_sentence_len // 2 - len(in_truncated))
            input_ids_truncated.append(in_truncated)

        return torch.LongTensor(input_ids_truncated)

    def ab_tokenized(pair_wise_instances):
        instances_a, instances_b = zip(*pair_wise_instances)

        tokenized_a = tokenizer(list(instances_a), add_special_tokens=False)
        tokenized_b = tokenizer(list(instances_b), add_special_tokens=False)

        tokenized_a = truncate_with_mentions(tokenized_a['input_ids'])
        positions_a = torch.arange(tokenized_a.shape[-1]).expand(tokenized_a.shape)
        tokenized_b = truncate_with_mentions(tokenized_b['input_ids'])
        positions_b = torch.arange(tokenized_b.shape[-1]).expand(tokenized_b.shape)

        tokenized_ab_ = torch.hstack((tokenized_a, tokenized_b))
        positions_ab = torch.hstack((positions_a, positions_b))

        tokenized_ab_dict = {'input_ids': tokenized_ab_,
                             'attention_mask': (tokenized_ab_ != tokenizer.pad_token_id),
                             'position_ids': positions_ab
                             }

        return tokenized_ab_dict

    if truncate:
        tokenized_ab = ab_tokenized(pairwise_bert_instances_ab)
        tokenized_ba = ab_tokenized(pairwise_bert_instances_ba)
    else:
        instances_ab = [' '.join(instance) for instance in pairwise_bert_instances_ab]
        instances_ba = [' '.join(instance) for instance in pairwise_bert_instances_ba]
        tokenized_ab = tokenizer(list(instances_ab), add_special_tokens=False, padding=True)

        tokenized_ab_input_ids = torch.LongTensor(tokenized_ab['input_ids'])

        tokenized_ab = {'input_ids': torch.LongTensor(tokenized_ab['input_ids']),
                         'attention_mask': torch.LongTensor(tokenized_ab['attention_mask']),
                         'position_ids': torch.arange(tokenized_ab_input_ids.shape[-1]).expand(tokenized_ab_input_ids.shape)}

        tokenized_ba = tokenizer(list(instances_ba), add_special_tokens=False, padding=True)
        tokenized_ba_input_ids = torch.LongTensor(tokenized_ba['input_ids'])
        tokenized_ba = {'input_ids': torch.LongTensor(tokenized_ba['input_ids']),
                        'attention_mask': torch.LongTensor(tokenized_ba['attention_mask']),
                        'position_ids': torch.arange(tokenized_ba_input_ids.shape[-1]).expand(tokenized_ba_input_ids.shape)}

    return tokenized_ab, tokenized_ba    
    

# Tokenize the test_transcripts here, similarly to how you did for train and dev sets
# You can use tokenize_props or a similar function, depending on how you need the data to be structured for testing

    
def train(train_pairs,
          train_labels,
          dev_pairs,
          dev_labels,
          parallel_model,
          proposition_map,
          working_folder,
          device,
          batch_size=16,
          n_iters=50,
          lr_lm=0.00001,
          lr_class=0.001,
          group=20,
          resultsFolder = ''):
    bce_loss = torch.nn.BCELoss()
    # mse_loss = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW([
        {'params': parallel_model.module.model.parameters(), 'lr': lr_lm},
        {'params': parallel_model.module.linear.parameters(), 'lr': lr_class}
    ])


    tokenizer = parallel_model.module.tokenizer

    # prepare data
    train_ab, train_ba = tokenize_props(tokenizer, train_pairs, proposition_map, parallel_model.module.end_id, max_sentence_len=512)
    dev_ab, dev_ba = tokenize_props(tokenizer, dev_pairs, proposition_map, parallel_model.module.end_id, max_sentence_len=512)
   
    #labels
    train_labels = torch.FloatTensor(train_labels)
    dev_labels = torch.LongTensor(dev_labels)
    print("train tensor size",train_ab['input_ids'].size())
    print("dev tensor size",dev_ab['input_ids'].size())
    print("train label size", len(train_labels))
    print("dev label size", len(dev_labels))
    train_loss = []
    
    
    #print('This is the pairs - ', train_ab)
    for n in range(n_iters):
        
        train_indices = list(range(len(train_pairs)))
        random.shuffle(train_indices)
        iteration_loss = 0.
        # new_batch_size = batching(len(train_indices), batch_size, len(device_ids))
        new_batch_size = batch_size
        for i in tqdm(range(0, len(train_indices), new_batch_size), desc='Training'):
            optimizer.zero_grad()
            batch_indices = train_indices[i: i + new_batch_size]

            scores_ab = forward_ab(parallel_model, train_ab, device, batch_indices, cosine_sim=False)
            scores_ba = forward_ab(parallel_model, train_ba, device, batch_indices, cosine_sim=False)

            batch_labels = train_labels[batch_indices].reshape((-1, 1)).to(device)

            scores_mean = (scores_ab + scores_ba) / 2

            loss = bce_loss(scores_mean, batch_labels)

            loss.backward()

            optimizer.step()

            iteration_loss += loss.item()

        print(f'Iteration {n} Loss:', iteration_loss / len(train_pairs))
        train_loss.append(iteration_loss / len(train_pairs))
        # iteration accuracy
        dev_scores_ab, dev_scores_ba = predict_with_XE(parallel_model, dev_ab, dev_ba, device, batch_size,cosine_sim=False)
        dev_predictions = (dev_scores_ab + dev_scores_ba)/2
        #print(dev_predictions)
        dev_predictions = dev_predictions > 0.5
        dev_predictions = torch.squeeze(dev_predictions)
        #print(dev_predictions)
        #print(dev_predictions, dev_labels)
        
        
        print("dev accuracy:", accuracy(dev_predictions, dev_labels))
        print("dev precision:", precision(dev_predictions, dev_labels))
        print("dev recall:", recall(dev_predictions, dev_labels))
        print("dev f1:", f1_score(dev_predictions, dev_labels))
        plt.plot(train_loss)
        plt.show()
     
        #You can choose to save the model
        # if n % 2 == 0:
        #     scorer_folder = working_folder + f'/XE_scorer/chk_{n}'
        #     if not os.path.exists(scorer_folder):
        #         os.makedirs(scorer_folder)
        #     model_path = scorer_folder + '/linear.chkpt'
        #     torch.save(parallel_model.module.linear.state_dict(), model_path)
        #     parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
        #     parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert')
        #     print(f'saved model at {n}')
        
    import json
    #function to save metrics
    def save_metrics_and_predictions(filename, metrics, predictions, labels):
    # Make sure this function only deals with Python data types, not PyTorch tensors
        with open(filename, 'w') as f:
            json.dump({'metrics': metrics, 'predictions': predictions, 'labels': labels}, f)
    final_accuracy = accuracy(dev_predictions, dev_labels)
    final_precision = precision(dev_predictions, dev_labels)
    final_recall = recall(dev_predictions, dev_labels)
    final_f1 = f1_score(dev_predictions, dev_labels)
    # Before calling the function, convert tensors to Python lists (or numbers for metrics)
    metrics = {
        'accuracy': final_accuracy.item() if torch.is_tensor(final_accuracy) else final_accuracy,
        'precision': final_precision.item() if torch.is_tensor(final_precision) else final_precision,
        'recall': final_recall.item() if torch.is_tensor(final_recall) else final_recall,
        'f1': final_f1.item() if torch.is_tensor(final_f1) else final_f1,
    }

    predictions = dev_predictions.cpu().tolist() if torch.is_tensor(dev_predictions) else dev_predictions
    labels = dev_labels.cpu().tolist() if torch.is_tensor(dev_labels) else dev_labels

    filename = f'trainDevMetrics/{resultsFolder}group{group}.json'
    save_metrics_and_predictions(filename, metrics, predictions, labels)

    #This creates the test dataset with only the positive pairs 
    def create_test_set(dev_pairs, dev_labels, proposition_map):
        positive_dev_pairs = [pair for pair, label in zip(dev_pairs, dev_labels) if label == 1]
        
        test_instances = []
        for pair in positive_dev_pairs:
            transcript = proposition_map[pair]['transcript'].replace("<m>", "").replace("</m>", "").strip()
            common_ground = proposition_map[pair]['common_ground']
            test_instances.append({'transcript': transcript, 'common_ground': common_ground})

        return test_instances

    #this just gets all of the positive pairs from the test set
    test_instances = create_test_set(dev_pairs, dev_labels, proposition_map) 

    # Create a DataFrame from the test instances
    test_df = pd.DataFrame(test_instances, columns=['transcript', 'common_ground'])
    test_df["Label"] = 1
    
    #get the list of all possible common grounds
    common_grounds_dataSet = pd.read_csv('/s/babbage/b/nobackup/nblancha/public-datasets/ilideep/XE/Data/goldenFiles/NormalizedList.csv')
    common_grounds = list(common_grounds_dataSet['Propositions'])
    
    new_rows = []
    all_cosine_rows = []
    parallel_model = parallel_model.to(device)
    evaluation_results = []
    genericCosine = False
    propsLost = 0
    #for each of the transctipt in the test dataset, get the transcript and generate the pruned possible common grounds. 
    for index, row in test_df.iterrows():
        #original_common_ground = row['common_ground'].replace("and", " , ") #raw common ground
        original_common_ground = row['common_ground'].replace("<m>", "").replace("</m>", "").strip()
        elements = extract_colors_and_numbers(row['transcript'].lower()) #The list of colors / weights in the transcript
        filtered_common_grounds = []
        filtered_common_grounds = [cg for cg in common_grounds if is_valid_common_ground(cg, elements)]

        if not filtered_common_grounds:  # If no match found, try individual color-number pairs
            filtered_common_grounds = [cg for cg in common_grounds if is_valid_individual_match(cg, elements)]  #If there is no match where only the mentioned colors and weights are present, get the individual combincations 
        if not is_proposition_present(original_common_ground, filtered_common_grounds):
            print('{group} did not have a proposition. \n')
            print('original' ,original_common_ground , '\n')
            print('transcirpt' ,row['transcript'].lower())
    
        
        print('Length of filtered Common grounds - ', len(filtered_common_grounds))
        #now get the cosine similarity between the current transcript in the test set and all possible common_grounds
        cosine_similarities = []
        for cg in filtered_common_grounds:
            
            if(genericCosine):
                input = parallel_model.module.tokenizer.encode_plus(row['transcript'].lower(), cg, add_special_tokens=True, return_tensors="pt")
                input_ids = input['input_ids'].to(device)
                attention_mask = input['attention_mask'].to(device)
                position_ids = torch.arange(input_ids.size(1)).unsqueeze(0).to(device)

                # Generate vector representations
                _, transcript_vec, common_ground_vec = parallel_model.module.generate_cls_arg_vectors(
                    input_ids, attention_mask, position_ids, None, None, None
                )

                # Calculate cosine similarity
                cosine_similarity = F.cosine_similarity(transcript_vec, common_ground_vec).item()
                cosine_similarities.append(cosine_similarity)
            else:
                # Tokenize and prepare inputs using the tokenizer from parallel_model
                cg_with_token = "<m>" + " " + cg + " "  + "</m>"
                trans_with_token = "<m>" + " "+ row['transcript'] +" " + "</m>"
                theIndividualDict = {
                    "transcript": trans_with_token,
                    "common_ground": cg_with_token # match[0] is the common ground text
                }
                #new_df = pd.DataFrame(theIndividualDict, columns=["transcript", "common_ground"])
                #indecies = new_df.index.to_list()
                proposition_map = {0: theIndividualDict} 
                proposition_ids = [0]
                tokenizer = parallel_model.module.tokenizer
                test_ab, test_ba = tokenize_props(tokenizer,proposition_ids,proposition_map,parallel_model.module.end_id ,max_sentence_len=512, truncate=True)    
                
                
                
                cosine_test_scores_ab, cosine_test_scores_ba = predict_with_XE(parallel_model, test_ab, test_ba, device, batch_size,cosine_sim=True)
                cosine_similarity = (cosine_test_scores_ab + cosine_test_scores_ba) /2
                cosine_similarities.append(cosine_similarity)
            

        # Select top 5 matches based on cosine similarity
        top_matches = sorted(zip(filtered_common_grounds, cosine_similarities), key=lambda x: x[1], reverse=True)[:5]
        
        if not top_matches:  # If top_matches is empty
            print(f"Transcript: {row['transcript'].lower()}")
            print("Filtered common grounds with no top matches:", filtered_common_grounds)
            break
        
     
        
         # For each top match, create a new row with the transcript and the common ground
        for match in top_matches:
            new_row = {
                "transcript": row['transcript'],
                "common_ground": match[0]  # match[0] is the common ground text
            }
            new_rows.append(new_row)
        for cg, cosine_similarity in zip(filtered_common_grounds, cosine_similarities):
            all_cosine_row = {
                "transcript": row['transcript'],
                "filtered_common_ground": cg,  # The filtered common ground text
                "cosine_similarity": cosine_similarity,  # The cosine similarity score
                "true_common_ground": row['common_ground']  # The true common ground from the original data
            }
            all_cosine_rows.append(all_cosine_row)
    
    all_cosine_rows_df = pd.DataFrame(all_cosine_rows, columns=["transcript", "filtered_common_ground", "cosine_similarity", "true_common_ground"])
    all_cosine_rows_df.to_csv(f'cosineScores/{resultsFolder}cosine_Scores{group}.csv') 
    new_df = pd.DataFrame(new_rows, columns=["transcript", "common_ground"])
    new_df.index.to_list()#the list of indicies in the dict that needs to be tokenized
    
    proposition_map_test = new_df.to_dict(orient='index') #make it into a dict
    proposition_map_test = add_special_tokens(proposition_map_test)    # add the special tokens to transcript and common ground 

    #call tokenize props here.
    tokenizer = parallel_model.module.tokenizer
    new_df.to_csv("test_set.csv") #sanity check
    
    
    test_ab, test_ba = tokenize_props(tokenizer, new_df.index.to_list(), proposition_map_test, parallel_model.module.end_id, max_sentence_len=512, truncate=True)    
    
    test_scores_ab, test_scores_ba = predict_with_XE(parallel_model, test_ab, test_ba, device, batch_size,cosine_sim=False)
    test_predictions = (test_scores_ab + test_scores_ba)/2
    new_df["scores"] = test_predictions #Get the raw scores as given by the cross Encoder
    test_predictions = test_predictions > 0.5
    test_predictions = torch.squeeze(test_predictions)
    print(test_predictions)
    test_predictions = test_predictions > 0.5
    test_predictions = torch.squeeze(test_predictions)
   
    actual_common_ground_map = test_df.set_index('transcript')['common_ground'].to_dict()
    new_df['actual_common_ground'] = new_df['transcript'].map(actual_common_ground_map)# Set transcript as index for easy lookup
    new_df['Group'] =  group
    new_df.to_csv(f'resultsTrainedCosineUpdates/{resultsFolder}{group}.csv')
    #print('Total Props Lost - ' , propsLostCosine)

    scorer_folder = '../' + working_folder + f'/{resultsFolder}/BERTXE_scorerOralce{group}/' 
    if not os.path.exists(scorer_folder):
        os.makedirs(scorer_folder)
    model_path = scorer_folder + '/linear.chkpt'
    torch.save(parallel_model.module.linear.state_dict(), model_path)
    parallel_model.module.model.save_pretrained(scorer_folder + '/bert')
    parallel_model.module.tokenizer.save_pretrained(scorer_folder + '/bert')
    return parallel_model

if __name__ == '__main__':
    resultsFolderArgs = str(sys.argv[1])
    print(resultsFolderArgs)
    model_name = str(sys.argv[2])
    train_prop_XE('ecb', model_name=model_name,resultsFolder = resultsFolderArgs)

