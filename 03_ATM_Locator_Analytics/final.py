import sys
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: python {__file__} train.tsv test.tsv")
        exit()
    train_path = sys.argv[1]
    test_path = sys.argv[2]

    train_df = shuffle(pd.read_table(train_path))
    test_df = pd.read_table(test_path)

    object_fea = train_df.select_dtypes(include=['object']).columns.tolist()
    ordinal_encoder = OrdinalEncoder()
    train_df[object_fea] = ordinal_encoder.fit_transform(train_df[object_fea]).astype(int)
    test_df[object_fea] = ordinal_encoder.transform(test_df[object_fea]).astype(int)

    # Regression
    y_train = train_df['revenue'].values
    x_train = train_df.drop('revenue', axis=1).values
    x_test = test_df[list(train_df.drop('revenue', axis=1).columns)]
    # y_test = test_df['revenue'].values          #

    rf_model = RandomForestRegressor()
    rf_model.fit(x_train, y_train)
    y_pred = rf_model.predict(x_test.values)

    y_pre_df = pd.DataFrame(y_pred, columns=['predicted_revenue'])
    y_pre_df.to_csv('output2.csv', index=False)


    # results_df = pd.DataFrame({'true_values': y_test, 'predicted_values': y_pred})
    # correlation = results_df['true_values'].corr(results_df['predicted_values'])
    # print(correlation)


    # Classify
    y_train2 = train_df['rating'].values
    x_train2 = train_df.drop('rating', axis=1).values
    x_test2 = test_df[list(train_df.drop('rating', axis=1).columns)]
    # y_test2 = test_df['rating'].values          #

    rf_model2 = RandomForestRegressor()
    rf_model2.fit(x_train2, y_train2)
    y_pred2 = rf_model2.predict(x_test2.values)
    y_pred2 = np.round(y_pred2).astype(int)

    # from sklearn.metrics import f1_score
    # f_score = f1_score(y_test2, y_pred2, average='weighted')
    # print(f_score)

    y_pre_df2 = pd.DataFrame(y_pred2, columns=['predicted_rating'])
    y_pre_df2.to_csv('output1.csv', index=False)




