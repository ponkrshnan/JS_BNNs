"""
Configuration file for the main code

"""
dataset = "cifar" #histo or cifar
run_no = 5  # 1 to 5
noise = 0.6 # For CIFAR data
loss = "JSG" #JSA, JSG or KL
num_outputs = 10 # 2 for histo and 10 for cifar
batch_size = 64
epochs = 50
learning_rate = 1e-4
num_samples = 1
weight_scale =0.1
rho_offset = -10
lamda = 1 # Used for JSA and JSG
alpha = 0.2249   #Used for JSA and JSG
sigma_pr = 0.1
out_dir = 'Experiments/'+dataset+'/'+loss+'/run_'+str(run_no)
if dataset == 'cifar':
    out_dir = out_dir +'/noise_' + str(noise)
model = "resnet18" 
