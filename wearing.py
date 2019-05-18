import pandas as pd
import torch as nn
import load_run_nets

my_net = load_run_nets.load_resnet(data_list=[], num_net=34, classifier=True, csv_file=False, num_labels=76)

print("Loading NN...")
my_net.load_state_dict(nn.load('./data/cnn_state', map_location='cpu'))

df_labels = pd.read_csv('./data/labels_marks')


def wearing_classification(img):
    try:
        assert img.shape == (3, 224, 224)
    except AssertionError:
        print('invalid input, shape incorrect\n')
        return None
    
    y_hat = prediction2classes(my_net.forward(nn.FloatTensor([img])))
    
    out_str = df_labels.to_dict()[str(y_hat)][0]
    return out_str


def prediction2classes(output_var):
    _, predicted = nn.max(output_var.data, 1)
    predicted.squeeze_()
    classes = predicted.tolist()
    return classes