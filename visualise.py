import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Function to plot the KDE for female vs male grades
def plot_sex_vs_grades(df):
    sns.kdeplot(df.loc[df['sex'] == 'F', 'G3'], label='F', color='#ff9999')
    sns.kdeplot(df.loc[df['sex'] == 'M', 'G3'], label='M', color='#004c99')

    plt.title('Female vs Male Grades (G3)')
    plt.xlabel('Final Grade (G3)')
    plt.legend()
    plt.show()

# Function to plot bar plots for Mother's and Father's Education Levels
def plot_education_vs_grades(df):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    a = sns.barplot(x=df['Medu'], y=df['G3'], data=df, hue=df['school'], palette=['#526D82', '#27374D'], errorbar=None, ax=ax[0])
    a.set_xlabel("Mother's Education Level")

    b = sns.barplot(x=df['Fedu'], y=df['G3'], data=df, hue=df['school'], palette=['#526D82', '#27374D'], errorbar=None, ax=ax[1])
    b.set_xlabel("Father's Education Level")

    plt.show()

# Function to plot Travel Time vs Grades for MS and GP schools
def plot_travel_time_vs_grades(df):
    avg_trv_MS = []
    avg_trv_GP = []

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    a = sns.barplot(x=df[df['school'] == "MS"]["traveltime"], y=df['G3'], data=df, errorbar=None, palette='colorblind', ax=ax[0])
    ax[0].text(0, 11.4, 'MS-School', size='x-large')
    a.set_xlabel('Travel Time')

    b = sns.barplot(x=df[df['school'] == "GP"]["traveltime"], y=df['G3'], data=df, errorbar=None, palette='colorblind', ax=ax[1])
    ax[1].text(-0.15, 11.6, 'GP-School', size='x-large')
    b.set_xlabel('Travel Time')

    plt.tight_layout()
    plt.show()

    # Calculate and print average grades for different travel times
    for i in range(4):
        avg_trv_MS.append(df.loc[(df['traveltime'] == i+1) & (df['school'] == 'MS'), 'G3'].mean())
        avg_trv_GP.append(df.loc[(df['traveltime'] == i+1) & (df['school'] == 'GP'), 'G3'].mean())

    avg_trv_MS = [x.round(2) if not np.isnan(x) else 0 for x in avg_trv_MS]
    avg_trv_GP = [x.round(2) if not np.isnan(x) else 0 for x in avg_trv_GP]

    print(f'Average grades for MS school travel time: {avg_trv_MS}')
    print(f'Average grades for GP school travel time: {avg_trv_GP}')

# Function to plot Study Time vs Grades for MS and GP schools
def plot_study_time_vs_grades(df):
    avg_stud_MS = []
    avg_stud_GP = []

    for i in range(4):
        avg_stud_MS.append(df.loc[(df['studytime'] == i+1) & (df['school'] == 'MS'), 'G3'].mean())
        avg_stud_GP.append(df.loc[(df['studytime'] == i+1) & (df['school'] == 'GP'), 'G3'].mean())

    avg_stud_MS = [x.round(2) if not np.isnan(x) else 0 for x in avg_stud_MS]
    avg_stud_GP = [x.round(2) if not np.isnan(x) else 0 for x in avg_stud_GP]

    print(f'Average grades for MS school study time: {avg_stud_MS}')
    print(f'Average grades for GP school study time: {avg_stud_GP}')

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    a = sns.barplot(x=df[df['school'] == "MS"]["studytime"], y=df['G3'], data=df, errorbar=None, palette='colorblind', ax=ax[0])
    ax[0].text(0, 12.5, 'MS-School', size='x-large')
    a.set_xlabel('Study Time')

    b = sns.barplot(x=df[df['school'] == "GP"]["studytime"], y=df['G3'], data=df, errorbar=None, palette='colorblind', ax=ax[1])
    ax[1].text(-0.15, 12, 'GP-School', size='x-large')
    b.set_xlabel('Study Time')

    for i in range(4):
        ax[0].text(i, 5, avg_stud_MS[i], ha='center', size='large')
        ax[1].text(i, 5, avg_stud_GP[i], ha='center', size='large')

    plt.tight_layout()
    plt.show()

# Function to plot the line graph of Absences vs Grades
def plot_absences_vs_grades(df):
    sns.lineplot(x=df['absences'], y=df['G3'], hue=df['school'], data=df, errorbar=('ci', False))
    plt.xlabel('Absence')
    plt.show()

# Function to plot Lineplot of G1 and G2 vs G3
def plot_grades_vs_grades(df):
    fig, ax = plt.subplots(1, 2)

    sns.lineplot(x=df['G1'], y=df['G3'], data=df, hue=df['school'], ax=ax[0], errorbar=('ci', False))
    sns.lineplot(x=df['G2'], y=df['G3'], data=df, hue=df['school'], ax=ax[1], errorbar=('ci', False))

    plt.tight_layout()
    plt.show()

# Main function to call the plotting functions
def main():
    # Load dataset
    df = pd.read_csv('student-mat.csv')  # Replace with the correct path to your dataset

    # Plot KDE for Female vs Male Grades
    plot_sex_vs_grades(df)

    # Plot Education vs Grades
    plot_education_vs_grades(df)

    # Plot Travel Time vs Grades for MS and GP
    plot_travel_time_vs_grades(df)

    # Plot Study Time vs Grades for MS and GP
    plot_study_time_vs_grades(df)

    # Plot Absences vs Grades
    plot_absences_vs_grades(df)

    # Plot G1 and G2 vs G3
    plot_grades_vs_grades(df)

if __name__ == "__main__":
    main()
