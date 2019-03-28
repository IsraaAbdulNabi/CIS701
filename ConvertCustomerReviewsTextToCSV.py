import pandas as pd

def convertText_CSV(txtFileName,csvFileName):
    data = []
    review = []
    f = open(txtFileName, "r")
    for x in f:
        review.append(int(x[9]))
        review.append(x[11:])
        data.append(review.copy())
        review.clear()
    df = pd.DataFrame(data, columns = ['StarRating_Label', 'CustomerReview_Data'])
    df.to_csv (csvFileName, index = None, header=True)

if __name__ == '__main__':
    convertText_CSV('train_data_sample1.txt','train_dat_sample1.csv')