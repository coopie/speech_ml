import os
import subprocess

number_of_subjects = 5

subjects_data = [filename.split('.')[0].split('_') for filename in os.listdir('corpora/RAVDESS') if filename.endswith('.wav')]

dummy_files = ["should_be_ignored.txt"]

for id, emotion, scenario, repitition in subjects_data:
    subject_number = int(id[len('RAVDESS'):])
    if subject_number <= number_of_subjects and scenario == 'kid' and repitition == '1' and emotion in ['happy', 'sad']:
        print subject_number
        dummy_files.append("_".join([id[len('RAVDESS'):], emotion, scenario, repitition]) + ".wav")


subprocess.call(['touch'] + map(lambda x: 'test/dummy_data/' + x, dummy_files))

print dummy_files
print len(dummy_files)
