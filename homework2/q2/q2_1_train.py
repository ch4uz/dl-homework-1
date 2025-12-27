import torch
import torch.optim as optim
from utils import masked_mse_loss, masked_spearman_correlation, load_rnacompete_data, plot
import q2_1_cnn as cnn_module
import q2_1_lstm as lstm_module

def train_network(model_class, model_name, protein='RBFOX1', epochs=20, batch_size=64, lr=0.001):
    print(f"\n--- Training {model_name} on {protein} ---")
    
    train_dset = load_rnacompete_data(protein, split='train')
    val_dset   = load_rnacompete_data(protein, split='val')
    test_dset  = load_rnacompete_data(protein, split='test')
    
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_dset, batch_size=batch_size, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_dset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)


    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        batch_loss_accum = 0
        
        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            
            optimizer.zero_grad()
            preds = model(x)
            
   
            loss = masked_mse_loss(preds, y, mask)
            
            loss.backward()
            optimizer.step()
            batch_loss_accum += loss.item()
        
        avg_train_loss = batch_loss_accum / len(train_loader)
        train_losses.append(avg_train_loss)


        model.eval()
        all_preds, all_targs, all_masks = [], [], []
        
        with torch.no_grad():
            for x, y, mask in val_loader:
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                preds = model(x)
                
                all_preds.append(preds)
                all_targs.append(y)
                all_masks.append(mask)
                
        vp = torch.cat(all_preds)
        vt = torch.cat(all_targs)
        vm = torch.cat(all_masks)
        
        val_loss = masked_mse_loss(vp, vt, vm).item()
        val_corr = masked_spearman_correlation(vp, vt, vm).item()
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Corr: {val_corr:.4f}")

    model.eval()
    test_preds, test_targs, test_masks = [], [], []
    with torch.no_grad():
        for x, y, mask in test_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            test_preds.append(model(x))
            test_targs.append(y)
            test_masks.append(mask)
            
    tp = torch.cat(test_preds)
    tt = torch.cat(test_targs)
    tm = torch.cat(test_masks)
    
    test_corr = masked_spearman_correlation(tp, tt, tm).item()
    print(f"--- Final Test Spearman Correlation: {test_corr:.4f} ---")
    
    plot(range(1, epochs+1), 
         {'Train Loss': train_losses, 'Val Loss': val_losses}, 
         filename=f"{model_name}_loss_plot.png")
    
    return test_corr


if __name__ == "__main__":
    EPOCHS = 20
    BATCH_SIZE = 64
    LR = 0.001

    cnn_score = train_network(cnn_module.RNA_CNN, "CNN_Model", 
                              epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)

    lstm_score = train_network(lstm_module.RNA_LSTM, "LSTM_Model", 
                               epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)

    print("\nSummary:")
    print(f"CNN Test Correlation: {cnn_score:.4f}")
    print(f"LSTM Test Correlation: {lstm_score:.4f}")