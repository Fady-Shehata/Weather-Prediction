import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


x_temp = np.arange(0, 11, 1)
x_cover = np.arange(0, 11, 1)
x_weather  = np.arange(0, 26, 1)

# Generate fuzzy membership functions
temp_coo = fuzz.trimf(x_temp, [0, 0, 5])
temp_war = fuzz.trimf(x_temp, [0, 5, 10])
temp_hot = fuzz.trimf(x_temp, [5, 10, 10])
cov_sun = fuzz.trimf(x_cover, [0, 0, 5])
cov_part = fuzz.trimf(x_cover, [0, 5, 10])
cov_over = fuzz.trimf(x_cover, [5, 10, 10])
weat_rain = fuzz.trimf(x_weather, [0, 0, 13])
weat_cle = fuzz.trimf(x_weather, [0, 13, 25])
weat_sno = fuzz.trimf(x_weather, [13, 25, 25])

# Visualize these universes and membership functions
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(x_temp, temp_coo, 'b', linewidth=1.5, label='Cool')
ax0.plot(x_temp, temp_war, 'g', linewidth=1.5, label='Warm')
ax0.plot(x_temp, temp_hot, 'r', linewidth=1.5, label='Hot')
ax0.set_title('Temperature')
ax0.legend()

ax1.plot(x_cover, cov_sun, 'b', linewidth=1.5, label='Sunny')
ax1.plot(x_cover, cov_part, 'g', linewidth=1.5, label='Partly')
ax1.plot(x_cover, cov_over, 'r', linewidth=1.5, label='Overcast')
ax1.set_title('Cover')
ax1.legend()

ax2.plot(x_weather, weat_rain, 'b', linewidth=1.5, label='Rainy')
ax2.plot(x_weather, weat_cle, 'g', linewidth=1.5, label='Clear')
ax2.plot(x_weather, weat_sno, 'r', linewidth=1.5, label='Snow')
ax2.set_title('Weather')
ax2.legend()


for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

temp_level_coo = fuzz.interp_membership(x_temp, temp_coo, 6.5)
temp_level_war = fuzz.interp_membership(x_temp, temp_war, 6.5)
temp_level_ho = fuzz.interp_membership(x_temp, temp_hot, 6.5)

cov_level_sun = fuzz.interp_membership(x_cover, cov_sun, 9.8)
cov_level_part = fuzz.interp_membership(x_cover, cov_part, 9.8)
cov_level_over = fuzz.interp_membership(x_cover, cov_over, 9.8)


active_rule1 = np.fmax(temp_level_coo, cov_level_sun)


weath_activation_rain = np.fmin(active_rule1, weat_rain) 

# For rule 2 we connect acceptable service to medium weatherping
weath_activation_clea = np.fmin(cov_level_part, weat_cle)

# For rule 3 we connect high service OR high food with high weatherping
active_rule3 = np.fmax(temp_level_ho, cov_level_over)
weath_activation_sno = np.fmin(active_rule3, weat_sno)
weath0 = np.zeros_like(x_weather)

# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(x_weather, weath0, weath_activation_rain, facecolor='b', alpha=0.7)
ax0.plot(x_weather, weat_rain, 'b', linewidth=0.5, linestyle='--', )
ax0.fill_between(x_weather, weath0, weath_activation_clea, facecolor='g', alpha=0.7)
ax0.plot(x_weather, weat_cle, 'g', linewidth=0.5, linestyle='--')
ax0.fill_between(x_weather, weath0, weath_activation_sno, facecolor='r', alpha=0.7)
ax0.plot(x_weather, weat_sno, 'r', linewidth=0.5, linestyle='--')
ax0.set_title('Output membership activity')

# Turn off top/right axes
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

aggregated = np.fmax(weath_activation_rain,
                     np.fmax(weath_activation_clea, weath_activation_sno))

# Calculate defuzzified result
weather = fuzz.defuzz(x_weather, aggregated, 'centroid')
weather_activation = fuzz.interp_membership(x_weather, aggregated, weather)  # for plot

# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(x_weather, weat_rain, 'b', linewidth=0.5, linestyle='--', )
ax0.plot(x_weather, weat_cle, 'g', linewidth=0.5, linestyle='--')
ax0.plot(x_weather, weat_sno, 'r', linewidth=0.5, linestyle='--')
ax0.fill_between(x_weather, weath0, aggregated, facecolor='Orange', alpha=0.7)
ax0.plot([weather, weather], [0, weather_activation], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('Aggregated membership and result (line)')

# Turn off top/right axes
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()