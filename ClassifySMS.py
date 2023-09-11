import cPickle

nb_detector_reloaded = cPickle.load(open('sms_spam_detector.pkl'))


def predictSpam(txt):
    return nb_detector_reloaded.predict([txt])[0], nb_detector_reloaded.predict_proba([txt])[0]

while True:
    sms = raw_input("Enter SMS text to classify: ")
    sham, prob = predictSpam(sms)
    if sham == 'ham':
        print('The SMS is not a spam (Confidence: ' + str(prob[0]) + ')')
    else:
        print('The SMS is a spam (Confidence:' + str(prob[1]) + ')')
