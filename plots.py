import matplotlib.pyplot as plt



def pairwise_plot(df, classifier_1 : str, classifier_2 : str, named_datasets = None):
    plt.scatter(df["classifier_1"],df["classifier_2"])
    plt.plot([0,1],[0,1],lw = 0.5)
    plt.title("{} vs {}".format(classifier_1,classifier_2))
    plt.xlabel(classifier_2.capital())
    plt.ylabel(classifier_1.capital())

    # 

def cd_diagram():
    pass


def 
