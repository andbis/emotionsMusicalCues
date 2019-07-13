#############################################
##major run script - cogsci2emotionandmusic##
#############################################

import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from library import *
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

data = pd.read_table('data/design_matrix.tab')
temp = data.iloc[:,[1,3,4,5,6]]
mode = pd.get_dummies(data.iloc[:,2])
melody = pd.get_dummies(data.iloc[:,-1]) 

data_mat = np.zeros((200,9))
feature_names = list(data.iloc[:,1:].columns)
feature_names.pop(1)
feature_names.pop(-1)
feature_names.append('Mode')
feature_names.append('MelodyS1') #
feature_names.append('MelodyS2')
feature_names.append('MelodyS3')
data_mat[:,:5] = temp.as_matrix()
data_mat[:,5] = mode.as_matrix()[:,0]
data_mat[:,6:] = melody.as_matrix()[:,:-1]
labels = pd.read_table('data/mean_emotion_ratings.tab')


scary_all = labels.iloc[:,1]
happy_all = labels.iloc[:,2]
sad_all = labels.iloc[:,3]
peaceful_all = labels.iloc[:,4]

y_all = [list(scary_all)] + [list(happy_all)] + [list(sad_all)] + [list(peaceful_all)]
y_array = np.array(y_all).T
label_names = ['Scary', 'Happy', 'Sad', 'Peaceful']

#Correlation between the four rate emotions
#Happy - sad, happy - scary, happy - peaceful
#sad - scary, sad - peaceful
#scary - peaceful 
print("Correlation between scary and happy: %f, with p-value: %f" % pearsonr(scary_all, happy_all))
print()
print("Correlation between sad and happy: %f, with p-value: %f" % pearsonr(sad_all, happy_all))
print()
print("Correlation between peaceful and happy: %f, with p-value: %f" % pearsonr(peaceful_all, happy_all))
print()
print("Correlation between scary and sad: %f, with p-value: %f" % pearsonr(scary_all, sad_all))
print()
print("Correlation between scary and peaceful: %f, with p-value: %f" % pearsonr(scary_all, peaceful_all))
print()
print("Correlation between sad and peaceful: %f, with p-value: %f" % pearsonr(sad_all, peaceful_all))
print()


#emotion scales with highest juded example for each emotion to see overall discrimination of the scales
scary_idx = int(scary_all.argsort()[::-1][:1])
happy_idx = int(happy_all.argsort()[::-1][:1])
sad_idx = int(sad_all.argsort()[::-1][:1])
peaceful_idx = int(peaceful_all.argsort()[::-1][:1])

highest_scored = labels.iloc[[scary_idx, happy_idx, sad_idx, peaceful_idx], 1:]
emotion_names = ['Scary', 'Happy', 'Sad', 'Peaceful']
highest_scored.index = emotion_names
Colour = 'r', 'g', 'b', 'y'
df_plot = pd.DataFrame(index=[a for a in range(16)], columns=['Emotion', 'Highest Rated', 'Mean-score', 'Colour'])
for idx, el in enumerate(highest_scored.iteritems()):
    indel = idx*4
    df_plot.iloc[indel:indel+4,0] = el[0]
    #scary
    df_plot.iloc[indel, 1] = emotion_names[0]
    df_plot.iloc[indel, 2] = el[1][0]
    df_plot.iloc[indel, 3] = Colour[0]
    #happy
    df_plot.iloc[indel+1, 1] = emotion_names[1]
    df_plot.iloc[indel+1, 2] = el[1][1]
    df_plot.iloc[indel+1, 3] = Colour[1]
    #sad
    df_plot.iloc[indel+2, 1] = emotion_names[2]
    df_plot.iloc[indel+2, 2] = el[1][2]
    df_plot.iloc[indel+2, 3] = Colour[2]
    #peaceful
    df_plot.iloc[indel+3, 1] = emotion_names[3]
    df_plot.iloc[indel+3, 2] = el[1][3]
    df_plot.iloc[indel+3, 3] = Colour[3]

sns.set(font_scale=1.6)
sns.set_style("darkgrid")

colorcodes = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00"]

fig, ax = plt.subplots(figsize=(10,10))

sns.stripplot(ax=ax, x='Emotion', y='Mean-score', hue ="Highest Rated", data=df_plot, jitter=True, palette=(colorcodes), size=15)
plt.savefig('emotion.png')
print('emotion.png is saved in current directory')

emotion_names = ['Scary', 'Happy', 'Sad', 'Peaceful']
fig = plt.figure(figsize=(10,18))

fig.subplots_adjust(wspace=0.2, hspace=0.3)#, top=None)

leg1 = mpatches.Patch(color='r')
leg2 = mpatches.Patch(color='g')
leg3 = mpatches.Patch(color='b')
leg4 = mpatches.Patch(color='y')

register_xs, register_ys = plotting_cue(data, 1, labels)
ax1 = fig.add_subplot(421)
ax1.plot(register_xs[0], register_ys[0], 'r-')
ax1.plot(register_xs[1], register_ys[1], 'g-')
ax1.plot(register_xs[2], register_ys[2], 'b-')
ax1.plot(register_xs[3], register_ys[3], 'y-')
ax1.axis([0,6.5, 1.5,4])
ax1.set_xticks=(register_xs[0])
ax1.set_xticklabels(["", '53','59','65','71','77','84'])
ax1.set_xlabel('MIDI Note')
fig.legend(handles=[leg1, leg2, leg3, leg4], labels=emotion_names, loc="upper center")
plt.title('Register')


mode_xs, mode_ys = plotting_cue(data, 2, labels)
ax1 = fig.add_subplot(422)
ax1.plot(mode_xs[0], mode_ys[0], 'r-')
ax1.plot(mode_xs[1], mode_ys[1], 'g-')
ax1.plot(mode_xs[2], mode_ys[2], 'b-')
ax1.plot(mode_xs[3], mode_ys[3], 'y-')
ax1.set_xticks=(1,2)
ax1.axis([0.8,2.2, 1.5,4])
ax1.set_xticklabels(["","Major","","","","", "Minor"], fontsize=12)
plt.title('Mode')

tempo_xs, tempo_ys = plotting_cue(data, 3, labels)
ax2 = fig.add_subplot(423)
ax2.plot(tempo_xs[0], tempo_ys[0], 'r-')
ax2.plot(tempo_xs[1], tempo_ys[1], 'g-')
ax2.plot(tempo_xs[2], tempo_ys[2], 'b-')
ax2.plot(tempo_xs[3], tempo_ys[3], 'y-')
ax2.axis([0.5,5.5, 1.5,4])
ax2.set_xticks(tempo_xs[0])
ax2.set_xticklabels(['1.2','2.0','2.8','4.4','6.0'])
ax2.set_xlabel('Notes per second')
plt.title('Tempo')

db_xs, db_ys = plotting_cue(data, 4, labels)
ax2 = fig.add_subplot(424)
ax2.plot(db_xs[0], db_ys[0], 'r-')
ax2.plot(db_xs[1], db_ys[1], 'g-')
ax2.plot(db_xs[2], db_ys[2], 'b-')
ax2.plot(db_xs[3], db_ys[3], 'y-')
ax2.axis([0.5,5.5, 1.5,4])
ax2.set_xticks(db_xs[0])
ax2.set_xticklabels(['-10','-5','0','+5','+10'])
ax2.set_xlabel('decibel')
plt.title('Soundlevel')

art_xs, art_ys = plotting_cue(data, 5, labels)
ax3 = fig.add_subplot(425)
ax3.plot(art_xs[0], art_ys[0], 'r-')
ax3.plot(art_xs[1], art_ys[1], 'g-')
ax3.plot(art_xs[2], art_ys[2], 'b-')
ax3.plot(art_xs[3], art_ys[3], 'y-')
ax3.axis([0.5,4.5, 1.5,4])
ax3.set_xticks(art_xs[0])
ax3.set_xticklabels(['1.0','.75','.5','.25'])
ax3.set_xlabel('Relative Duration')
plt.title('Articulation')

tim_xs, tim_ys = plotting_cue(data, 6, labels)
ax3 = fig.add_subplot(426)
ax3.plot(tim_xs[0], tim_ys[0], 'r-')
ax3.plot(tim_xs[1], tim_ys[1], 'g-')
ax3.plot(tim_xs[2], tim_ys[2], 'b-')
ax3.plot(tim_xs[3], tim_ys[3], 'y-')
ax3.axis([0.5,3.5, 1.5,4])
ax3.set_xticks(tim_xs[0])
ax3.set_xticklabels(['Flute','French Horn','Trumpet'])
plt.title('Timbre')

mel_xs, mel_ys = plotting_cue(data, 7, labels)
ax4 = fig.add_subplot(427)
ax4.plot(mel_xs[0], mel_ys[0], 'r-')
ax4.plot(mel_xs[1], mel_ys[1], 'g-')
ax4.plot(mel_xs[2], mel_ys[2], 'b-')
ax4.plot(mel_xs[3], mel_ys[3], 'y-')
ax4.axis([0.5,4.5, 1.5,4])
ax4.set_xticks(mel_xs[0])
ax4.set_xticklabels(['Peaceful','Happy','Scary','Sad'])
plt.title('Melody - Structure')
plt.savefig('features.png')
print('features.png saved in current directory\n')

#SR^2 - squared semipartial correlations
run_data = data_mat
run_features = feature_names

sr2s = []
for idx, emotion in enumerate(emotion_names):
    spc = semi_partial_correlation_squared(run_data, y_all[idx])
    sr2_dic = {}
    for feature, sr2 in zip(run_features, spc):
        sr2_dic[feature] = sr2
    sr2s.append((emotion, sr2_dic))

for i in sr2s:
	print("Squared semi-partial correlation for emotion",i[0])
	print(i[1])
	print()

#Neural network
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

#preparing train and test sets 
n_fold = 10
cv = KFold(n_fold, random_state=442)
xtraining = []
xtesting = []
ytraining = []
ytesting = []
idx = 0
for train, test in cv.split(data_mat):
    trainx, testx, trainy, testy = data_mat[train], data_mat[test], y_array[train], y_array[test]
    scaler = MinMaxScaler(feature_range=(-1,1)).fit(trainx)
    X_train = scaler.transform(trainx)
    X_test = scaler.transform(testx)
    xtraining.append(X_train)
    xtesting.append(X_test)
    ytraining.append(trainy)
    ytesting.append(testy)
    idx += 1


#settings for #crossvalidation
epok = 100
verb = 0
scaler = MinMaxScaler(feature_range=(-1,1)).fit(data_mat)
norm_data = scaler.transform(data_mat)
mean_emotion_r2 = {}
r2_container = []
median_cont = {}

for idx, emotion in enumerate(emotion_names):
	k = 0
	emotion_r2 = []
	for train_x, test_x, train_y, test_y in zip(xtraining, xtesting, ytraining, ytesting):
		print('Learning in CV', k, 'for emotion', emotion)
		model = Sequential()
		model.add(Dense(12, activation='relu', input_shape=(norm_data.shape[1],)))
		model.add(Dense(8, activation='relu'))
		model.add(Dense(1, activation='relu'))  
		model.compile(loss='mean_squared_error',
		              optimizer='adam',
		              metrics=['mean_squared_error'])
		model.fit(train_x, train_y[:,idx], epochs=epok, batch_size=8, verbose=verb)

		predicted = model.predict(test_x)
		emotion_r2.append(r2_adjusted(test_y[:,idx], predicted))
		k += 1
	mean_emotion_r2[emotion] = sum(emotion_r2)/len(emotion_r2)
	median_cont[emotion] = np.median(emotion_r2)
	r2_container.append(emotion_r2)
print('10-CV model results mean:', mean_emotion_r2)
print('median', median_cont)

###########################

#settings for full set model
epok = 100
verb = 0
scaler = MinMaxScaler(feature_range=(-1,1)).fit(data_mat)
norm_data = scaler.transform(data_mat)
mean_emotion_r2 = {}
r2_container = []
median_cont = {}
#emotion_names = ["Scary"]
k = 0
overall_pred = []
for idx, emotion in enumerate(emotion_names):
	emotion_r2 = []
	print('Learning for emotion', emotion)
	model = Sequential()
	model.add(Dense(12, activation='relu', input_shape=(norm_data.shape[1],)))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='relu'))  
	model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mean_squared_error'])
	model.fit(norm_data, y_all[idx], epochs=epok, batch_size=8, verbose=verb)

	predicted = model.predict(norm_data)
	emotion_r2.append(r2_adjusted(y_all[idx], predicted))
	k += 1
	    #plt.show(hinton(model.get_weights()[0]))
	r2_container.append(emotion_r2)
	mean_emotion_r2[emotion] = sum(emotion_r2)/len(emotion_r2)
	median_cont[emotion] = np.median(emotion_r2)
	overall_pred.append(predicted)

print('Full model results:', mean_emotion_r2)
print('median', median_cont)

#pca
labels['Class'] = labels.iloc[:,1:].idxmax(axis=1)
Colour = ['r', 'g', 'b', 'y']
emotion_names = ['Scary', 'Happy', 'Sad', 'Peaceful']
jk = labels.iloc[:,-1]
c = [Colour[emotion_names.index(a)] for a in jk]


e_values, e_vectors, centred, feature_means = pca(y_array)
two = e_vectors[:2,:]
transformed2 = np.dot(two, centred.T).T

y_2array = np.array(overall_pred).T[0]
e_values2, e_vectors2, centred2, feature_means2 = pca(y_2array)
transformed22 = np.dot(two, centred2.T).T
#https://stackoverflow.com/questions/47391702/matplotlib-making-a-colored-markers-legend-from-scratch
blue_o = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                          markersize=10, label='Sad True')
red_o = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                          markersize=10, label='Scary True')
green_o = mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                          markersize=10, label='Happy True')
yellow_o = mlines.Line2D([], [], color='yellow', marker='o', linestyle='None',
                          markersize=10, label='Peaceful True')
blue_x = mlines.Line2D([], [], color='blue', marker='x', linestyle='None',
                          markersize=10, label='Sad Predicted')
red_x = mlines.Line2D([], [], color='red', marker='x', linestyle='None',
                          markersize=10, label='Scary Predicted')
green_x = mlines.Line2D([], [], color='green', marker='x', linestyle='None',
                          markersize=10, label='Happy Predicted')
yellow_x = mlines.Line2D([], [], color='yellow', marker='x', linestyle='None',
                          markersize=10, label='Peaceful Predicted')
red = mpatches.Patch(color='r')
green = mpatches.Patch(color='g')
blue = mpatches.Patch(color='b')
yellow = mpatches.Patch(color='y')
hands = [blue_o, red_o, green_o, yellow_o, blue_x, red_x, green_x, yellow_x]

fig = plt.figure(figsize=(10,10))
plt.scatter(transformed2[:,0], transformed2[:,1], color=c, s=30, marker='o', alpha=.3)
plt.scatter(transformed22[:,0], transformed22[:,1], color=c, s=30, marker='x', alpha=1)
plt.axis('equal')
plt.ylabel('PC2')
plt.xlabel('PC1')
plt.title('True and predicted labels projected onto first two eigenvectors of true labels')
plt.legend(handles=hands)#, labels=['blue_star', 'red_square', 'purple_triangle'])
plt.savefig('predictionsvtrue.png')

print('predictionsvtrue.png saved in current directory\n All computations done\n')
