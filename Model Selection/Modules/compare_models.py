import matplotlib.pyplot as plt
import numpy as np

labels = ['Linear Reg.','NN','Clustering','Classification']
x = np.arange(len(labels))
train_rmse = [9186, 9616, 5443, 8014]
test_rmse = [7865, 7378, 7798, 7097]
train_mae = [5871, 6207, 5256, 5910]
test_mae = [5858, 5162, 6113, 5268]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
#rects1 = ax.bar(x - width/2, train_rmse, width, label='RMSE')
#rects2 = ax.bar(x + width/2, train_mae, width, label='MAE')
rects1 = ax.bar(x - width/2, test_rmse, width, label='RMSE')
rects2 = ax.bar(x + width/2, test_mae, width, label='MAE')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
#ax.set_title('Estimation Scores Compared - Train')
ax.set_title('Estimation Scores Compared - Test')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()