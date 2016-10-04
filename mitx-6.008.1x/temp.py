# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 20:03:44 2016

@author: piyushmaheshwari
"""

from simpsons_paradox_data import *;

print (joint_prob_table[gender_mapping['female'], 
                                              department_mapping['C'], 
admission_mapping['admitted']])


joint_prob_gender_admission = joint_prob_table.sum (axis = 1)

female_only = joint_prob_gender_admission [gender_mapping['female']]
prob_admission_given_female = female_only / np.sum (female_only)
prob_admission_given_female_dict = dict (zip (admission_labels, prob_admission_given_female))
print (prob_admission_given_female_dict)

male_only = joint_prob_gender_admission [gender_mapping['male']]
prob_admission_given_male = male_only / np.sum (male_only)
prob_admission_given_male_dict = dict (zip (admission_labels, prob_admission_given_male))
print (prob_admission_given_male_dict)

admitted_only = joint_prob_gender_admission[:, admission_mapping['admitted']]
prob_gender_given_admitted = admitted_only / np.sum (admitted_only)
prob_gender_given_admitted_dict = dict (zip(gender_labels, prob_gender_given_admitted))
print (prob_gender_given_admitted_dict)

for dept in ['A', 'B', 'C', 'D', 'E', 'F']:
    print ("Results for dept :" + dept)
    
    admitted_given_female_A = joint_prob_table [gender_mapping['female'], department_mapping[dept]]
    prob_admitted_given_female_A = admitted_given_female_A / np.sum (admitted_given_female_A)
    prob_admitted_given_female_A_dict = dict (zip(admission_labels, prob_admitted_given_female_A))
    print (prob_admitted_given_female_A_dict)
    
    admitted_given_male_A = joint_prob_table [gender_mapping['male'], department_mapping[dept]]
    prob_admitted_given_male_A = admitted_given_male_A / np.sum (admitted_given_male_A)
    prob_admitted_given_male_A_dict = dict (zip(admission_labels, prob_admitted_given_male_A))
    print (prob_admitted_given_male_A_dict)

sum = 0
for i in [1,2,4]:
    for j in [1,3]:
        sum += i*i + j*j

print (sum)
                                