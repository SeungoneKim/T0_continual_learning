import argparse
import wandb
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Trainer, DataCollatorForSeq2Seq
from dataset import CT0Dataset_train, CT0Dataset_eval
from custom_metrics import computeBleu, computeSari, computeRouge, computeConstrain, computeHaiku, computeBERTScore, FirstWordSim, Clf

# Set Argument Parser
parser = argparse.ArgumentParser()
# Training hyperparameters
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--train_batch_size', type=int, default=2)
parser.add_argument('--val_batch_size',type=int, default=2)
# Optimizer hyperparameters
parser.add_argument('--init_lr',type=float, default=1e-4)
parser.add_argument('--warm_up',type=int, default=0)
parser.add_argument('--weight_decay',type=float, default=0)
parser.add_argument('--decay_epoch',type=int, default=0)
parser.add_argument('--adam_beta1',type=float, default=0.9)
parser.add_argument('--adam_beta2',type=float, default=0.999)
parser.add_argument('--adam_eps',type=float, default=1e-12)
parser.add_argument('--dropout_rate',type=float, default=0.1)
# Tokenizer hyperparameters
parser.add_argument('--encoder_max_len', type=int, default=512)
parser.add_argument('--decoder_max_len', type=int, default=128)
# Data hyperparameters
parser.add_argument('--phase', type=str, default='phase1')
parser.add_argument('--eval_only',type=bool, default=False)
parser.add_argument('--eval_dataset',type=str)
# Checkpoint directory hyperparameters
parser.add_argument('--model_cache',type=str)
parser.add_argument('--weight_path',type=str)
args = parser.parse_args()
args.model_cache = "/home/seungone/cache/T0_original_weights"
args.weight_path = f"./{args.phase}_weight_path"

# Set GPU
print('######################################################################')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())
print(torch.cuda.get_device_name())
print('######################################################################')

# Start WANDB Log (Set Logging API)
wandb.init(project="continualT0", reinit=True, entity='lklab_kaist')

# Set tokenizer
tokenizer = AutoTokenizer.from_pretrained('bigscience/T0_3B')

# Set data
train_data = CT0Dataset_train(tokenizer, args.encoder_max_len, args.decoder_max_len, args.phase, device)
eval_data = {}
if args.phase=='phase1':
    eval_data['wiki_auto'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'wiki_auto', device)
    #eval_data['asset'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'asset', device)
elif args.phase=='phase2':
    eval_data['wiki_auto'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'wiki_auto', device)
    eval_data['asset'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'asset', device)
    eval_data['gigaword'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'gigaword', device)
elif args.phase=='phase3':
    eval_data['wiki_auto'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'wiki_auto', device)
    eval_data['asset'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'asset', device)
    eval_data['gigaword'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'gigaword', device)
    eval_data['haiku'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'haiku', device)
elif args.phase=='phase4':
    eval_data['wiki_auto'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'wiki_auto', device)
    eval_data['asset'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'asset', device)
    eval_data['gigaword'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'gigaword', device)
    eval_data['haiku'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'haiku', device)
    eval_data['covid_qa'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'covid_qa', device)
elif args.phase=='phase5':
    eval_data['wiki_auto'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'wiki_auto', device)
    eval_data['asset'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'asset', device)
    eval_data['gigaword'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'gigaword', device)
    eval_data['haiku'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'haiku', device)
    eval_data['covid_qa'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'covid_qa', device)
    eval_data['eli5'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'eli5', device)
elif args.phase=='phase6':
    eval_data['wiki_auto'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'wiki_auto', device)
    eval_data['asset'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'asset', device)
    eval_data['gigaword'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'gigaword', device)
    eval_data['haiku'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'haiku', device)
    eval_data['covid_qa'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'covid_qa', device)
    eval_data['eli5'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'eli5', device)
    eval_data['emdb'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'emdb', device)
elif args.phase=='phase7':
    eval_data['wiki_auto'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'wiki_auto', device)
    eval_data['asset'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'asset', device)
    eval_data['gigaword'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'gigaword', device)
    eval_data['haiku'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'haiku', device)
    eval_data['covid_qa'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'covid_qa', device)
    eval_data['eli5'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'eli5', device)
    eval_data['emdb'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'emdb', device)
    eval_data['eSNLI'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'eSNLI', device)
elif args.phase=='phase8':
    eval_data['wiki_auto'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'wiki_auto', device)
    eval_data['asset'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'asset', device)
    eval_data['gigaword'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'gigaword', device)
    eval_data['haiku'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'haiku', device)
    eval_data['covid_qa'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'covid_qa', device)
    eval_data['eli5'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'eli5', device)
    eval_data['emdb'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'emdb', device)
    eval_data['eSNLI'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'eSNLI', device)
    eval_data['twst'] = CT0Dataset_eval(tokenizer, args.encoder_max_len, args.decoder_max_len, 'twst', device)

# Set model
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_cache,local_files_only=True)

# Set DataCollator
#collator = DataCollatorForSeq2Seq(tokenizer,model,padding=True)

# Set evaluation
def compute_metrics(eval_preds):
    logits, labels, srcs = eval_preds
    predictions = np.argmax(logits, axis=-1)
    score = {}
    bleu_score = computeBleu(predictions, labels)
    sari_score = computeSari(predictions, labels, srcs)
    rouge_score = computeRouge(predictions, labels)
    constrain_score_start = computeConstrain(predictions, labels, src_info={"constrain_type":"start"})
    haiku_score = computeHaiku(predictions, labels)
    bertscore_score = computeBERTScore(predictions, labels)
    firstwordsim_score = FirstWordSim(predictions, labels)
    clf_score = Clf(predictions, labels)

    score = bleu_score+sari_score+rouge_score+constrain_score_start+haiku_score+bertscore_score+firstwordsim_score+clf_score
    return metric.compute(predictions=predictions, references=labels)



finetune_args = Seq2SeqTrainingArguments(
    output_dir = args.weight_path,
    overwrite_output_dir = True,
    do_train=True,
    do_eval=True,
    evaluation_strategy='epoch',
    per_device_train_batch_size = args.train_batch_size,
    per_device_eval_batch_size = args.val_batch_size,
    learning_rate=args.init_lr,
    weight_decay=args.weight_decay,
    adam_beta1=args.adam_beta1,
    adam_beta2=args.adam_beta2,
    adam_epsilon=args.adam_eps,
    num_train_epochs=args.epoch,
    max_grad_norm=0.1,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    lr_scheduler_type='polynomial',
    warmup_steps= args.warm_up,
    logging_strategy="epoch",
    save_strategy= "epoch",
    save_total_limit=5,
    fp16=True,
    seed = 42,
    include_inputs_for_metrics = True,
    remove_unused_columns = False,
    group_by_length=True,
    load_best_model_at_end=False,
    predict_with_generate=True,
    prediction_loss_only=False,
    generation_max_length=args.decoder_max_len,
    generation_num_beams=1,
    greater_is_better=False,
    report_to = 'wandb',
)

trainer = Trainer(
    model = model,
    args = finetune_args,
    train_dataset = train_data,
    eval_dataset = eval_data,
    tokenizer = tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# Save final weights
finetune_trainer.save_model(args.weight_path)