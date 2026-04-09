def two_stage_predict(cls, reg, applicant_df):

    # Stage 1
    proba = cls.predict_proba(applicant_df)
    preds = cls.predict(applicant_df)
    print("Approved : ",preds)
    print("Proba : ", type(proba), proba)
    print("Preds : ", type(preds), preds)

    result = []

    approved = int(preds[0])
    approved_prob = float(proba[0,1])

    # approved : True/False
    # approved_prob : 0 - 1
    # reg_pred : number/None

    # Stage 2
    loan_amount_predicted = None
    if approved == 1:
        print("Now in Stage 2 of model predict : ",applicant_df.shape)
        applicant_df = applicant_df.drop('loan_amount',axis=1)
        print("Dropping loan_amount column")
        print("New dimensions of DataFrame for Regression are : ",applicant_df.shape)
        loan_amount_predicted = float(reg.predict(applicant_df)[0])

    result.append({
        "approved" : approved,
        "approved_prob" : approved_prob,
        "reg_prediction" : loan_amount_predicted
    })

    return result