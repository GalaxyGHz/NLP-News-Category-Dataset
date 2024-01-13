# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

torch.cuda.empty_cache()

dataset = pd.read_json(r'data/preprocess_data.json', lines=True)

x_train, x_test, y_train, y_test = train_test_split(dataset['clean_text'], dataset['category'], test_size=0.2, random_state=42)

x_train = [x for x in x_train]
x_test = [x for x in x_test]

y_train = [x for x in y_train]
y_test = [x for x in y_test]

# label_encoder = LabelEncoder()
# y_train = torch.Tensor(label_encoder.fit_transform(np.array(y_train).reshape(-1, 1)))
# y_test = torch.Tensor(label_encoder.fit_transform(np.array(y_test).reshape(-1, 1)))
label_encoder = OneHotEncoder()
y_train = torch.Tensor(label_encoder.fit_transform(np.array(y_train).reshape(-1, 1)).todense())
y_test = torch.Tensor(label_encoder.fit_transform(np.array(y_test).reshape(-1, 1)).todense())

# x_train = x_train[:50]
# y_train = y_train[:50]

# %%
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

class MyDataset(Dataset):
  def __init__(self, data, label):
    self.data = data
    self.label = label

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    item = self.data[idx]
    target = self.label[idx]
    return {'input_ids': item[0], 'attention_mask': item[1], 'label': target}

x_train = tokenizer(x_train, max_length=50, padding='longest', truncation=True, return_tensors='pt')
x_train = torch.stack((x_train['input_ids'], x_train['attention_mask']), dim=1)

x_test = tokenizer(x_test, max_length=50, padding='longest', truncation=True, return_tensors='pt')
x_test = torch.stack((x_test['input_ids'], x_test['attention_mask']), dim=1)

batch_size = 50

train_dataset = MyDataset(x_train, y_train)
test_dataset = MyDataset(x_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# %%
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=15)

from torch.optim import Adam
optimizer = Adam(model.parameters(), lr=5e-5)


from transformers import get_scheduler
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)

# %%
from tqdm.auto import tqdm

def test_dataset_accuracy(model, dataset, device):
    dataset_len = len(dataset)
    accuracy = 0

    for batch in tqdm(dataset):
        torch.cuda.empty_cache()
        inputs = batch['input_ids'].to(device)
        attention = batch['attention_mask'].to(device)
        ground_truth = batch['label'].to(device)

        pred = model(input_ids=inputs, attention_mask=attention)
        pred = torch.nn.functional.softmax(pred.logits, dim=1)

        acc = int((torch.argmax(pred, dim=1) == torch.argmax(ground_truth, dim=1)).sum()) / len(batch["label"])
        accuracy += acc

    return accuracy / dataset_len

# %%
torch.cuda.empty_cache()

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        torch.cuda.empty_cache()
        optimizer.zero_grad()

        inputs = batch['input_ids'].to(device)
        attention = batch['attention_mask'].to(device)
        ground_truth = batch['label'].to(device)

        pred = model(input_ids=inputs, attention_mask=attention)
        pred = torch.nn.functional.softmax(pred.logits, dim=1)

        # train_acc = int((torch.argmax(pred, dim=1) == torch.argmax(ground_truth, dim=1)).sum()) / batch["label"]

        loss = torch.nn.functional.cross_entropy(pred, ground_truth)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        progress_bar.update(1)
    test_acc = test_dataset_accuracy(model, test_dataloader, device)
    print(test_acc)

pt_save_directory = "./our_transformer"
tokenizer.save_pretrained(pt_save_directory)
model.save_pretrained(pt_save_directory)



