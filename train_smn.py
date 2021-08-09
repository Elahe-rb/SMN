import torch
import math
import preprocess_data_smn

#train for each epoch
def train(model, loss_fn, optimizer, rows, batch_size, epoch, num_epochs, vocab, max_utt_num, max_utt_length, device, uids_rows, cluster_ids):

    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    total_acc = 0
    losses = []

    num_iters = math.ceil(len(rows) / batch_size)
    log_interval = math.ceil(num_iters / 5)

    for batch in range(num_iters):
        cs, rs, ys = preprocess_data_smn.process_train_data(rows, batch, batch_size, vocab, device, False, uids_rows, cluster_ids)

        optimizer.zero_grad()

        output = model(cs,rs)
        loss = loss_fn(output,ys)

        losses.append(loss.item)
        total_loss += loss.item() * ys.size(0)
        pred = output >= 0.5
        num_correct = (pred == ys.byte()).sum().item()
        total_acc += num_correct


        loss.backward()
        optimizer.step()

        description = 'Train: [{}/{}][{}/{}] curr loss: {:.3f}, Loss: {:.3f}, Acc: {:.3f}'.format(
            epoch + 1, num_epochs,
            batch + 1, num_iters,
            loss.item(),
            total_loss / (batch_size * batch + output.size(0)),
            total_acc / (batch_size * batch + output.size(0)))
        if batch % log_interval == 0:
            print(description)

    return model, losses

