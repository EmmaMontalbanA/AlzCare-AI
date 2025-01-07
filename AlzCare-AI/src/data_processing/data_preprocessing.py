# Load and preprocess dataset
data = df  
data = data.drop(columns=['DoctorInCharge'])

# Separate features and target
X = data.drop(columns=['Diagnosis'])
y = data['Diagnosis']

# Label encoding for categorical features
categorical_cols = ['Gender', 'Ethnicity', 'Smoking', 'EducationLevel', 'FamilyHistoryAlzheimers',
                    'CardiovascularDisease', 'Diabetes', 'Depression', 'HeadInjury', 'Hypertension',
                    'MemoryComplaints', 'BehavioralProblems', 'Confusion', 'Disorientation',
                    'PersonalityChanges', 'DifficultyCompletingTasks', 'Forgetfulness']

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Feature Engineering
# Health Risk Factors
X['HealthScore'] = (X['BMI'] + X['CholesterolTotal'] + X['CholesterolLDL'] - X['CholesterolHDL']) / 4
X['BP_Ratio'] = X['SystolicBP'] / X['DiastolicBP']

# Lifestyle Factors
X['LifestyleScore'] = (X['PhysicalActivity'] + X['DietQuality'] + X['SleepQuality']) / 3
X['SmokingAlcoholInteraction'] = X['Smoking'] * X['AlcoholConsumption']

# Medical History Aggregates
X['ChronicConditionsCount'] = (X['CardiovascularDisease'] +
                                X['Diabetes'] +
                                X['Hypertension'])

# Cognitive and Functional Assessment
X['CognitiveDeclineScore'] = (X['MMSE'] + X['FunctionalAssessment']) / 2
X['MemoryBehaviorIssuesCount'] = (X['MemoryComplaints'] + X['BehavioralProblems'])

# Interaction Features
X['Age_BMI_Interaction'] = X['Age'] * X['BMI']
X['Age_CholesterolInteraction'] = X['Age'] * X['CholesterolTotal']

# Lifestyle and Health Ratios
X['BMILifestyleRatio'] = X['BMI'] / (X['PhysicalActivity'] + X['DietQuality'] + X['SleepQuality'])

# Encoded Features
bins = [0, 30, 50, 70, 100]
labels = ['Young', 'Middle-aged', 'Senior', 'Elderly']
X['AgeGroup'] = pd.cut(X['Age'], bins=bins, labels=labels)

# Cholesterol Ratios
X['CholesterolLDL_HDL_Ratio'] = X['CholesterolLDL'] / X['CholesterolHDL']
X['CholesterolTriglycerides_Ratio'] = X['CholesterolTriglycerides'] / X['CholesterolTotal']

# Select only the engineered features
engineered_features = [
    'HealthScore', 'BP_Ratio', 'LifestyleScore', 'SmokingAlcoholInteraction',
    'ChronicConditionsCount', 'CognitiveDeclineScore', 'MemoryBehaviorIssuesCount',
    'Age_BMI_Interaction', 'Age_CholesterolInteraction', 'BMILifestyleRatio',
    'AgeGroup', 'CholesterolLDL_HDL_Ratio', 'CholesterolTriglycerides_Ratio'
]

X_engineered = X[engineered_features]

# Save the engineered features dataset
X_engineered.to_csv('X_engineered_features.csv', index=False)

# If you want to include the target variable
X_engineered_with_target = X_engineered.copy()
X_engineered_with_target['Diagnosis'] = y
X_engineered_with_target.to_csv('X_engineered_features_with_target.csv', index=False)
