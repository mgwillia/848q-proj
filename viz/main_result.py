import matplotlib.pyplot as plt
import numpy as np

cmap = plt.get_cmap('Set1')
colormap = np.array(list(cmap.colors))

log_lines = []
with open('../logs/matt_job_guess.out.33815', 'r') as read_file:
    log_lines = read_file.readlines()

loss_list, question_list, matches_list = [], [], []
for line in log_lines[1:1+(93*50)]:
    if 'Epoch Num' in line:
        parts = line.split(',')
        loss = float(parts[2].split(':')[1])
        correct_questions = float(parts[3].split(':')[1])
        correct_matches = float(parts[4].split(':')[1])
        loss_list.append(loss)
        question_list.append(correct_questions)
        matches_list.append(correct_matches)

losses = np.array(loss_list)
questions = np.array(question_list)
matches = np.array(matches_list)

loss_scale = matches.max() / losses.max()
#losses = losses * loss_scale

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Iterations')
ax1.set_ylabel('# Matches per Iteration')
ax1.plot(questions, label='Question-Question Matching', color=colormap[0])
ax1.plot(matches, label='Question-Context Matching', color=colormap[1])
 
# Adding Twin Axes to plot using dataset_2
ax2 = ax1.twinx()

ax2.set_ylabel('Loss')
ax2.plot(losses, label='Loss', color=colormap[2])

f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]
handles = [f("s", colormap[i]) for i in range(3)]
plt.legend(handles,
          ['Question-Question Matching', 'Question-Context Matching', 'Loss'],
          scatterpoints=1,
          loc='center right',
          ncol=1)

plt.savefig('main_result.png', bbox_inches='tight')
