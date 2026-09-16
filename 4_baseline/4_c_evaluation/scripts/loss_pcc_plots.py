import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def pcc_plot(df, figname, single):
    # ======================================
    # FONT setting
    # ======================================
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 14

    # ======================================
    # VIRIDIS COLORS palette selection
    # ======================================
    viridis = plt.cm.viridis
    train_color = viridis(0.2)
    val_color = viridis(0.8)

    if single:
        best_train_pcc_epoch = df.loc[df["train_pcc"].idxmax(), "epoch"]
        best_val_pcc_epoch = df.loc[df["val_pcc"].idxmax(), "epoch"]

        best_train_pcc = df["train_pcc"].max()
        best_val_pcc = df["val_pcc"].max()

        errorbar = None

    else:
        mean_pcc = df.groupby("epoch")[["train_pcc","val_pcc"]].mean()

        best_train_pcc_epoch = mean_pcc["train_pcc"].idxmax()
        best_val_pcc_epoch = mean_pcc["val_pcc"].idxmax()

        best_train_pcc = mean_pcc.loc[best_train_pcc_epoch,"train_pcc"]
        best_val_pcc = mean_pcc.loc[best_val_pcc_epoch,"val_pcc"]

        errorbar = "sd"

    # ======================================
    # LOSS/PCC PLOT 
    # ======================================
    fig, ax = plt.subplots(figsize=(16,12))

    sns.lineplot(
        data=df,
        x="epoch",
        y="train_pcc",
        errorbar=errorbar,
        color=train_color,
        linewidth=1,
        label = "Training PCC",
        ax=ax
    )

    sns.lineplot(
        data=df,
        x="epoch",
        y="val_pcc",
        errorbar=errorbar,
        color=val_color,
        linewidth=1,
        label = "Validation PCC",
        ax=ax
    )

    # ======================================
    # MINIMUM TRAIN LOSS / MAXIMUM TRAIN PCC 
    # ======================================
    ax.axvline(
            best_train_pcc_epoch,
            color=train_color,
            linestyle='--',
            linewidth=2,
            alpha=0.9,
            label=f'Max train PCC (ep {best_train_pcc_epoch})'
        )
    
    # ======================================
    # MINIMUM VALIDATION LOSS / PCC
    # ======================================
    ax.axvline(
            best_val_pcc_epoch,
            color=val_color,
            linestyle=':',
            linewidth=2.5,
            alpha=0.9,
            label=f'Max val PCC (ep {best_val_pcc_epoch})'
            )

    ax.axhline(
            y=best_train_pcc,
            color='red',        # colore
            linestyle='--',     # '-', '--', '-.', ':'
            linewidth=1,        # spessore
            alpha=0.8,          # trasparenza
            label='Max train PCC'
        )

    ax.axhline(
            y=best_val_pcc,
            color='red',
            linestyle=':',
            linewidth=1,
            alpha=0.8,
            label='Max val PCC'
        )
            
    # opzionale: evidenzia i punti minimi
    ax.scatter(best_train_pcc_epoch,
                best_train_pcc,
                color=train_color,
                s=20,
                zorder=5)

    ax.scatter(best_val_pcc_epoch,
                best_val_pcc,
                color=val_color,
                s=20,
                zorder=5)
    
    # ======================================
    # STILE
    # ======================================
    ax.set_xlabel('Epoch')

    ax.set_ylabel('PCC')


    ax.set_title(
                "Training and Validation PCCs"   
                )

    ax.grid(linestyle='--',
            alpha=0.3)

    ax.legend(frameon=True,
            loc='upper left',
            fontsize=8,
            title='Losses',
            title_fontsize=10)

    sns.despine()

    plt.tight_layout()

    # ======================================
    # HIGH QUALITY PNG
    # ======================================

    plt.savefig(f'../output/{figname}.png',
                dpi=600,
                bbox_inches='tight',
                facecolor='white')
    plt.close()

def loss_plot(df, figname, single):
    # ======================================
    # FONT setting
    # ======================================
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 14

    # ======================================
    # VIRIDIS COLORS palette selection
    # ======================================
    viridis = plt.cm.viridis
    train_color = viridis(0.2)
    val_color = viridis(0.8)

    if single:
        best_train_loss_epoch = df.loc[df["train_loss"].idxmin(), "epoch"]
        best_val_loss_epoch = df.loc[df["val_loss"].idxmin(), "epoch"]

        best_train_loss = df["train_loss"].min()
        best_val_loss = df["val_loss"].min()

        errorbar = None

    else:
        mean_loss = df.groupby("epoch")[["train_loss","val_loss"]].mean()

        best_train_loss_epoch = mean_loss["train_loss"].idxmin()
        best_val_loss_epoch = mean_loss["val_loss"].idxmin()

        best_train_loss = mean_loss.loc[best_train_loss_epoch,"train_loss"]
        best_val_loss = mean_loss.loc[best_val_loss_epoch,"val_loss"]

        errorbar = "sd"

    # ======================================
    # LOSS/PCC PLOT 
    # ======================================
    fig, ax = plt.subplots(figsize=(16,12))

    sns.lineplot(
        data=df,
        x="epoch",
        y="train_loss",
        errorbar=errorbar,
        color=train_color,
        linewidth=1,
        label = "Training Loss",
        ax=ax
    )

    sns.lineplot(
        data=df,
        x="epoch",
        y="val_loss",
        errorbar=errorbar,
        color=val_color,
        linewidth=1,
        label = "Validation LOSS",
        ax=ax
    )

    # ======================================
    # MINIMUM TRAIN LOSS / MAXIMUM TRAIN PCC 
    # ======================================
    ax.axvline(
            best_train_loss_epoch,
            color=train_color,
            linestyle='--',
            linewidth=2,
            alpha=0.9,
            label=f'Min train Loss (ep {best_train_loss_epoch})'
        )
    
    # ======================================
    # MINIMUM VALIDATION LOSS / PCC
    # ======================================
    ax.axvline(
            best_val_loss_epoch,
            color=val_color,
            linestyle=':',
            linewidth=2.5,
            alpha=0.9,
            label=f'Min val loss (ep {best_val_loss_epoch})'
            )

    ax.axhline(
            y=best_train_loss,
            color='red',        # colore
            linestyle='--',     # '-', '--', '-.', ':'
            linewidth=1,        # spessore
            alpha=0.8,          # trasparenza
            label='Min train Loss'
        )

    ax.axhline(
            y=best_val_loss,
            color='red',
            linestyle=':',
            linewidth=1,
            alpha=0.8,
            label='Min val Loss'
        )
            
    # opzionale: evidenzia i punti minimi
    ax.scatter(best_train_loss_epoch,
                best_train_loss,
                color=train_color,
                s=20,
                zorder=5)

    ax.scatter(best_val_loss_epoch,
                best_val_loss,
                color=val_color,
                s=20,
                zorder=5)
    
    # ======================================
    # STILE
    # ======================================
    ax.set_xlabel('Epoch')

    ax.set_ylabel('Loss')


    ax.set_title(
                "Training and Validation Loss"   
                )

    ax.grid(linestyle='--',
            alpha=0.3)

    ax.legend(frameon=True,
            loc='upper left',
            fontsize=8,
            title='Losses',
            title_fontsize=10)

    sns.despine()

    plt.tight_layout()

    # ======================================
    # HIGH QUALITY PNG
    # ======================================

    plt.savefig(f'../output/{figname}.png',
                dpi=600,
                bbox_inches='tight',
                facecolor='white')
    plt.close()

"""
# Initialize data frame to store 'history_run_{}.csv' files
val = pd.DataFrame({'epoch': [],'run': [],'train_loss': [],'val_loss': [],'train_pcc': [],'val_pcc': []})

for i in range(20):
#for i in range(500):
    # [128, 64, 32]	relu	0.7	MSE
    df = pd.read_csv(f'../../4_b_mlp/output/grid_search/relu_MSE_128_64_32_0.7/history_run_{i+1}.csv', sep=',')
    figname = f'loss_curve_run_{i+1}'
    loss_plot(df, figname, single = True)
    figname = f'pcc_curve_run_{i+1}'
    pcc_plot(df,figname, single = True)
    # Creates column storing run index
    #run = [i+1]*df.shape[0]
    #df.insert(1, 'run', run)
    # Adding run related df to general df
    #val = pd.concat([val, df], ignore_index=True)
"""