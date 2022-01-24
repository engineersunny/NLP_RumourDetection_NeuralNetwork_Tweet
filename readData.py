import json
import jsonlines
from anytree import Node, RenderTree

def readX(filepath):
    mLines = []
    with jsonlines.open(filepath) as f:
        for line in f.iter():  # 1 line contains the original twt and re-twts

            aLine = []
            for twt in line:  # twt and rtwts
                d = {}
                d['curr_id'] = twt['id_str']
                d['parent_id'] = twt['in_reply_to_status_id_str']
                d['text'] = twt['text']
                d['user_info'] = twt['user']
                d['created'] = twt['created_at']
                aLine.append(d)
            mLines.append(aLine)
    twtData = mLines
    return twtData

def readY(filepath):
    f = open(filepath)
    data = json.load(f)  # dictionary

    train_label = []
    for i in data:
        if data[i] == 'non-rumour':
            train_label.append(1)
        elif data[i] == 'rumour':
            train_label.append(0)
    f.close()
    return train_label

def selectRoot(twtData):
    #pick root tweet only
    xtrain_root = []
    for line in twtData:
        root_id = line[0]['curr_id']

        text = ""
        for twt in line:
            id = twt['parent_id']
            if id == root_id: #only child of root
                text += ' ' + twt['text']
            if id is None: #root
                text += twt['text'] + ' ' +str(len(line)-1) #feature : length of retweet
            else: pass

        xtrain_root.append(text)

    return xtrain_root

def selectRoot_wo_ftengineering(twtData):
    xtrain_root = []
    for line in twtData:

        text = ""
        for twt in line:
            id = twt['parent_id']
            if id is None:  # root
                text += twt['text'] + ' ' + str(len(line) - 1) + ' ' + twt['created'].split(' ')[3][:2]  # feature : length of retweet / posting time(hr)

        xtrain_root.append(text)

    return xtrain_root

def openfile():
    twtData = readX('./data/train.data.jsonl')
    xvalid = readX('./data/dev.data.jsonl')
    xtest = readX('./data/test.data.jsonl')
    xcovid = readX('./data/covid.data.jsonl')
    #############################################################################
    train_label = readY('./data/train.label.json')
    yvalid = readY('./data/dev.label.json')
    #############################################################################
    #pick root tweet only
    xtrain_root = selectRoot_wo_ftengineering(twtData)
    xvalid_root = selectRoot_wo_ftengineering(xvalid)
    xtest_root = selectRoot_wo_ftengineering(xtest)
    xcovid_root = selectRoot_wo_ftengineering(xcovid)

    # xtrain,ytrain, xvalid, yvalid, xtrain
    return xtrain_root, train_label, xvalid_root, yvalid, xtest_root, xcovid_root


def pickID(twtData):
    xtrain_root = []
    for line in twtData:
        for twt in line:
            id = twt['parent_id']
            if id is None:  # root
                xtrain_root.append(twt['curr_id'])
    return xtrain_root

def getID():
    twtData = readX('./data/train.data.jsonl')
    xvalid = readX('./data/dev.data.jsonl')
    xtest = readX('./data/test.data.jsonl')
    xcovid = readX('./data/test.data.jsonl')
    xtrainID = pickID(twtData)
    xvalidID = pickID(xvalid)
    xtestID = pickID(xtest)
    xcovidID = pickID(xcovid)
    return xtrainID,xvalidID,xtestID,xcovidID

