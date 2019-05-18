import pandas as pd
import torch as nn
import load_run_nets

df_labels = pd.read_csv('./data/labels_marks')

print("Loading NN...")
my_net = load_run_nets.load_resnet(data_list=[], num_net=34, classifier=True, csv_file=False, num_labels=76)
my_net.load_state_dict(nn.load('./data/cnn_state_10', map_location='cpu'))
my_net = my_net.eval()


def wearing_classification(img):
    """
    Функция, получающая на вход и возвращающая
    предсказанную категорию
    """
    try:
        assert img.shape == (3, 224, 224)
    except AssertionError:
        print('invalid input, shape incorrect\n')
        return None
    y_hat = prediction2classes(my_net.forward(nn.FloatTensor([img/255])))
    out_str = df_labels.to_dict()[str(y_hat)][0]
    return out_str


def prediction2classes(output_var):
    _, predicted = nn.max(output_var.data, 1)
    predicted.squeeze_()
    classes = predicted.tolist()
    return classes
