import subprocess



hyparam = {
    'mrpc':{'bs':32,'epochs':30,'lr':3e-5,'max_len':512,'wd':0.01}, # 3.7k
    'stsb':{'bs':32,'epochs':30,'lr':1e-5,'max_len':512,'wd':0.01}, # 7k
    'cola':{'bs':32,'epochs':30,'lr':3e-5,'max_len':512,'wd':0.01}, # 8.5k
    'rte':{'bs':32,'epochs':30,'lr':1e-5,'max_len':512,'wd':0.01},  
    ############################################################################
    'sst2':{'bs':32,'epochs':30,'lr':1e-5,'max_len':512,'wd':0.01},# 67k 
    'qnli':{'bs':32,'epochs':30,'lr':1e-5,'max_len':512,'wd':0.01},# 105k
    'qqp':{'bs':32,'epochs':30,'lr':1e-5,'max_len':512,'wd':0.01}, # 364k
    'mnli':{'bs':32,'epochs':30,'lr':1e-5,'max_len':512,'wd':0.01}, # 393k
}

methods = ['adamw']


for task_name in hyparam.keys():
    for alpha in alpha_list:
        hy = hyparam[task_name]
        max_len , bs, lr ,  wd,epochs  = hy['max_len'],hy['bs'],hy['lr'],hy['wd'],hy['epochs']
        model_name_or_path = "../pretrain-weights/roberta-base/"
        data_dir = "./data/glue"
        
        num_train_epochs = str(epochs)
        per_device_train_batch_size = str(bs)
        per_device_eval_batch_size = str(bs)
        weight_decay = str(wd)
        lr = str(lr)
        report_to = "wandb"

        for method in methods:
            output_dir = f"./outputs/{task_name}_full_finetune_{method}"

            command = [
                "python", "run_glue.py",
                "--model_name_or_path", model_name_or_path,
                "--task_name", task_name,
                "--data_dir", data_dir,
                "--output_dir", output_dir,
                "--num_train_epochs", num_train_epochs,
                "--per_device_train_batch_size", per_device_train_batch_size,
                "--per_device_eval_batch_size", per_device_eval_batch_size,
                "--learning_rate", str(lr),
                "--weight_decay", weight_decay,
                "--method", method,
                "--with_tracking",
                "--report_to", report_to,
            ]

            subprocess.run(command)