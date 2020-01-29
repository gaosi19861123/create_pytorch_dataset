
class config:

    abs_data_path = "C:/Users/dso-s.gao/Desktop/signate" #データ保存場所
    train_path = "/dtc_train/"
    train_ano = "/dtc_train_annotations/dtc_train_annotations/"

    test_path = "/dtc_test/"

    dict_category = {
        'Car': 1,
        'Bicycle': 2,
        'Pedestrian': 3,
        'Signal': 4,
        'Signs': 5,
        'Truck': 6,
        'Bus': 7,
        'SVehicle': 8,
        'Motorbike': 9,
        'Train': 10,
        } #カテゴリのデータ

    number_class = 11 #include backgroud
    lr = 0.005 #学習率

opt = config() 

opt.abs_data_path


