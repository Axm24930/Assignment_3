import matplotlib.pyplot as plt
# Data to plot
prog_languages = 'Java', 'Python', 'PHP', 'JavaScript', 'C#', 'C++'
Popularity = [22.2, 17.6, 8.8, 8, 7.7, 6.7]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
# Exploding the first slice
explode = (0.1, 0, 0, 0,0,0)
# Plot
plt.pie(Popularity, explode=explode, labels=prog_languages, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()
