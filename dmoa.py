import datasets
import json
import time
import random
import torch
import gc
import numpy as np
import copy
import os
from transformers import AutoModelForCausalLM, AutoTokenizer


############################### MoA inputs #####################################
n=4 # number of LLMs in the MoA network
M=2 # number of layers (0 layer means no MoA)
k=2 # number of additional proposer LLMs (minimum 1; maximum n-1)
N=10 # number of prompts to be evaluated from the evaluation dataset, depending on the compute available

print(f"\nmoa_multi_prop: n={n}, M={M}, k={k}, N={N}")



############################### Models are loaded in this part #################


model_dict = {
    "model1_name" : "Qwen/Qwen1.5-72B-Chat",
    
    "model2_name" : "meta-llama/Meta-Llama-3-70B-Instruct",
    
    "model3_name" : "mistralai/Mixtral-8x22B-Instruct-v0.1",
    
    "model4_name" : "databricks/dbrx-instruct",

}

model_name = model_dict["model1_name"]
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_name,local_files_only=True)

############################### LLM function is defined here ###################


def LLM(i, input_prompt):
    global model_name
    global model
    global tokenizer

    if model_name != model_dict["model"+str(i)+"_name"]:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model_name = model_dict["model"+str(i)+"_name"]
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name,local_files_only=True)

    messages = [{"role": "user", "content": input_prompt}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to('cuda')
    input_length = inputs.shape[1]
    outputs = model.generate(inputs, max_new_tokens=1000, do_sample=False)
    text = tokenizer.batch_decode(outputs[:, input_length:],skip_special_tokens=True)[0]
    return text



############################## Prompts arrivals from dataset here ##############


eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]

counter=0
# additional_prompt=" Give short and precise answer."
additional_prompt=""
agg_prompt=" You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability. Do not add any additional comments about how you created these responses. Just synthesize these responses as instructed."

node_list = list(range(1, n + 1)) # list of nodes 
lambda_0=1 #total prompt generation rate



prompt_response=[]
dictionary_list=[]

T=0
for example in eval_set:
    
    counter=counter+1
    if counter > N:
        break
    print(counter)
    
    T=T+np.random.exponential(1/lambda_0)
    
    i = random.choice(node_list)
    temp_dict = {}
    temp_dict = {
        "prompt" : example["instruction"],
        "dataset": example["dataset"],
        "prompt_number" : counter,
        "user_number" : i,
        "arrival_time" : T,
    }
    
    if M > 0:
        for m in range(0,M+1):
            if m < M:
                proposer_list = [i]
                rest_list = node_list[:i-1]+node_list[i:n+1]
                j_list = random.sample(rest_list,k)
                proposer_list = [i] + j_list
                proposer_inf={"proposer_list" : proposer_list}
                for entry in proposer_list:
                    proposer_inf[entry] = ["", 0]
                temp_dict[f"layer{m+1}_proposers"] = proposer_inf
            if m == M:
                proposer_list = [i]
                proposer_inf={"proposer_list" : proposer_list}
                for entry in proposer_list:
                    proposer_inf[entry] = ["", 0]
                temp_dict[f"layer{m+1}_proposers"] = proposer_inf
    else:
        temp_dict["layer1_proposers"] = {"proposer_list" : [i], i : ["", 0]}
    prompt_response.append(temp_dict.copy())

    


for m in range(M+1):
    
    if M > 0:
        if m == 0:
            for i in node_list:
                for item in prompt_response:
                    if i in item[f"layer{m+1}_proposers"]["proposer_list"]:
                        start_time=time.time()
                        response=LLM(i,item["prompt"])
                        end_time=time.time()
                        inf_time=end_time-start_time
                        item[f"layer{m+1}_proposers"][i]=[response,inf_time]
        else:
            for i in node_list:
                for item in prompt_response:
                    if i in item[f"layer{m+1}_proposers"]["proposer_list"]:
                        prev_responses=""
                        indx=1
                        for proposer in item[f"layer{m}_proposers"]["proposer_list"]:
                            prev_responses = prev_responses+"\n\nResponse-"+str(indx)+": "+item[f"layer{m}_proposers"][proposer][0]
                            indx=indx+1
                        start_time=time.time()
                        response=LLM(i,item["prompt"]+additional_prompt+agg_prompt+prev_responses)
                        end_time=time.time()
                        inf_time=end_time-start_time
                        item[f"layer{m+1}_proposers"][i]=[response,inf_time]
                        
    else:
        for i in node_list:
            for item in prompt_response:
                if i in item["layer1_proposers"]["proposer_list"]:
                    start_time=time.time()
                    response=LLM(i,item["prompt"])
                    end_time=time.time()
                    inf_time=end_time-start_time
                    item["layer1_proposers"][i]=[response,inf_time]

    
    




for item in prompt_response:
    
    final_response = item[f"layer{M+1}_proposers"][item["user_number"]][0]
    
    dictionary = {
    	"instruction": item["prompt"],
    	"dataset": item["dataset"],
    	"output": final_response,
    	"generator": f"moa_multi_prop_n{n}_M{M}_k{k}_N{N}",
    	"datasplit": "eval",
    }
    
    dictionary_list.append(dictionary.copy())
    
    

json_object = json.dumps(dictionary_list, indent=4)

with open(f"outputs_MoA_multi_prop_latency_n{n}_M{M}_k{k}_N{N}.json", "w") as outfile:
	outfile.write(json_object)

json_object = json.dumps(prompt_response, indent=4)

with open(f"prompt_response_multi_prop_n{n}_M{M}_k{k}_N{N}", "w") as outfile:
	outfile.write(json_object)






#################### max/avg queue-size and average final response time ########

packet_list=[]
for item in prompt_response:
    packet = {
        "prompt_number" : item["prompt_number"],
        "prompt_user" : item["user_number"],
        "arrival_time" : item["arrival_time"],
        "destinations" : [],
        "inf_times" : [],
    }
    for m in range(M+1):
        packet["destinations"].append(item[f"layer{m+1}_proposers"]["proposer_list"])
        inference_durations = []
        for i in item[f"layer{m+1}_proposers"]["proposer_list"]:
            inference_durations.append(item[f"layer{m+1}_proposers"][i][1])
        packet["inf_times"].append(inference_durations.copy())
    packet_list.append(packet.copy())
    

    
init_packet_list = copy.deepcopy(packet_list)


final_packet_list = []
device = {}
for i in range(n):
    device[i+1] = {}
    device[f"device{i+1}_time"] = 0
    device[f"queue{i+1}_size"] = 0
    device[f"queue{i+1}_avg_size"] = 0


while len(packet_list) != 0:
    
    
    packet = packet_list[0]
    destinations=packet["destinations"][0]
    temp_var=packet["arrival_time"]
    
    for i in destinations:
        device[i]=packet
    packet_list.remove(packet)
    
    temp_packet_list=[]
    for count in range(len(destinations)):
        temp=device[destinations[count]]
        device[destinations[count]]={}
        device[f"device{destinations[count]}_time"] = max(device[f"device{destinations[count]}_time"], temp_var)
        temp["arrival_time"]=device[f"device{destinations[count]}_time"]+packet["inf_times"][0][count]
        old_clock_time = copy.deepcopy(device[f"device{destinations[count]}_time"])
        device[f"device{destinations[count]}_time"] = copy.deepcopy(temp["arrival_time"])
        new_clock_time = copy.deepcopy(device[f"device{destinations[count]}_time"])
        clock_time_diff = new_clock_time - old_clock_time
        temp_packet_list.append(temp.copy())
        
        
        #average queue-size calculation
        element = temp.copy()
        el_list = packet_list.copy()
        el_list.append(element)
        el_list = sorted(el_list, key = lambda x: x["arrival_time"])
        el_indx = el_list.index(element)
        device[f"queue{destinations[count]}_size"] = 0
        for q_el_indx in range(el_indx):
            q_el = packet_list[q_el_indx]
            if destinations[count] in q_el["destinations"][0]:
                device[f"queue{destinations[count]}_size"] += 1
        device[f"queue{destinations[count]}_avg_size"] += device[f"queue{destinations[count]}_size"]*clock_time_diff  #cumulative sum
    
    
    temp_packet=max(temp_packet_list, key = lambda x: x["arrival_time"])
    temp_packet["destinations"].remove(temp_packet["destinations"][0])
    temp_packet["inf_times"].remove(temp_packet["inf_times"][0])
    
    if temp_packet["destinations"] == []:
        final_packet_list.append(temp_packet.copy())
    else:
        packet_list.append(temp_packet.copy())
        packet_list = sorted(packet_list, key = lambda x: x["arrival_time"])


    

sum = 0
for i in range(len(final_packet_list)):
    sum+=final_packet_list[i]["arrival_time"]-init_packet_list[i]["arrival_time"]
sum=sum/len(final_packet_list)

final_system_time = final_packet_list[-1]["arrival_time"]
for i in range(n):
    device[f"queue{i+1}_avg_size"] = device[f"queue{i+1}_avg_size"]/final_system_time   #time average
q_avg_size = 0
for i in range(n):
    q_avg_size += device[f"queue{i+1}_avg_size"]/n    #ensemble average

print(f"\nmoa_multi_prop: n={n}, M={M}, k={k}, N={N}")
print(f"\naverage inference time = {sum} seconds")
print(f"\naverage queue-size = {q_avg_size}")

