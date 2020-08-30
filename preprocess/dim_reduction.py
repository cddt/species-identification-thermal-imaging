from sklearn import manifold
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_dim_reduction(model, layer):
    # create a model to take inputs and spit out intermediate layer
    intermediate_layer_model = Model(inputs=model.input, 
                                     #outputs=model.get_layer(layer_name).output) #select layer by name
                                     outputs=model.layers[layer].output) #or by index
    intermediate_output = intermediate_layer_model.predict(X_train)
    
    # perform tsne
    tsne = manifold.TSNE(n_components=2).fit_transform(intermediate_output)
    
    # plot tsne
    plt.figure(figsize=(12,8))
    sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white', 'axes.edgecolor': 'black'})
    b = sns.scatterplot(x=tsne[:,0], y= tsne[:,1],
        hue=y_train_raw,
        palette=sns.color_palette("husl", 13),
        legend="full",
        alpha=0.5)
    b.legend(frameon=False, title = None, fontsize=12)
    b.axes.set_title("t-SNE Dimensionality Reduction",fontsize=20)
    b.tick_params(labelsize=15)
    sns.despine()
    plt.savefig("/home/cddt/data-space/COMPSCI760/temp/tsne.png")
    
    # perform PCA
    scaled_intermediate_output = StandardScaler().fit_transform(intermediate_output)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(scaled_intermediate_output)
    print(pca.explained_variance_ratio_)
    
    # plot PCA
    plt.figure(figsize=(12,8))
    sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white', 'axes.edgecolor': 'black'})

    b = sns.scatterplot(x=principalComponents[:,0], y= principalComponents[:,1],
        hue=y_train_raw,
        palette=sns.color_palette("husl", 13),
        legend="full",
        alpha=0.5)
    b.legend(frameon=False, title = None, fontsize=12)
    b.axes.set_title("PCA Dimensionality Reduction",fontsize=20)
    b.set_xlabel("PCA-1 (9.8%)", fontsize=18)
    b.set_ylabel("PCA-2 (5.0%)", fontsize=18)
    b.tick_params(labelsize=15)
    sns.despine()
    plt.savefig("/home/cddt/data-space/COMPSCI760/temp/pca.png")
