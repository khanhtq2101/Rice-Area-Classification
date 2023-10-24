import torch
from sklearn.metrics import accuracy_score
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, optimizer, criterion, trainloader, writer, n_epochs = 1):
  loss_curve = []
  n_batches = len(trainloader)
  base_epoch = model.trained_epochs

  for epoch in range(n_epochs):
    for i, data in enumerate(trainloader):
      optimizer.zero_grad()

      series, label = data

      series = series.to(device)      
      label = label.to(device)
      
      output = model(series)
      
      loss = criterion(output, label)
      loss.backward()
      optimizer.step()

      loss = loss.item()
      loss_curve.append(loss)
      writer.add_scalar('Train/Loss', loss, i + (base_epoch + epoch)*n_batches)
      if i % 100 == 0:
        print("Epoch", epoch, "Batch (", i, "/", n_batches, ") \n",
              "\t Loss:", round(loss, 2))
  
  model.trained_epochs += n_epochs
        
  return loss_curve

def test(model, dataloader):
  n_batches = len(dataloader)
  pred = torch.tensor([]).to(device)
  ground_truth = torch.tensor([]).to(device)

  for i, data in enumerate(dataloader):
    series, label = data

    series = series.to(device)
    label = label.to(device)

    output = model(series)
    output_label = torch.argmax(output, dim= 1)

    pred = torch.cat((pred, output_label))
    ground_truth = torch.cat((ground_truth, label))

    if i % 200 == 0:
      print("Batch (", i, "/", n_batches, ")")

      running_pred = pred.cpu().numpy()
      running_ground_truth = ground_truth.cpu().numpy()
      acc = accuracy_score(running_pred, running_ground_truth)
      print(acc)

  pred = pred.cpu().numpy()
  ground_truth = ground_truth.cpu().numpy()

  return pred, ground_truth