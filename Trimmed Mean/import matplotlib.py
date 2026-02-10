import matplotlib.pyplot as plt

# Data
labels = ['Data Tampering', 'Data Privacy in EV Networks', 'Impersonation Attack', 
          'Side-Channel Attacks', 'SQL Injection', 'Battery Manipulation Attack', 
          'Malware Injection via IVI System', 'Data Poisoning Attack']
sizes = [10, 15, 10, 12, 8, 7, 13, 25]
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#ffb3e6','#c2c2f0','#ffb366','#ff6666']

# Plotting
plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.

# Title

plt.show()
