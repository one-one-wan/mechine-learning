import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import operator
from matplotlib.font_manager import FontProperties

# 设置字体路径，以解决中文显示问题
font = FontProperties(fname='C:/Windows/Fonts/simhei.ttf')  # 根据实际路径设置字体文件

# 特征字典，定义每个特征的取值范围
featureDic = {
    '色泽': ['浅白', '青绿', '乌黑'],
    '根蒂': ['硬挺', '蜷缩', '稍蜷'],
    '敲声': ['沉闷', '浊响', '清脆'],
    '纹理': ['清晰', '模糊', '稍糊'],
    '脐部': ['凹陷', '平坦', '稍凹'],
    '触感': ['硬滑', '软粘']
}


# 获取数据集
def getDataSet():
    dataSet = [
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '好瓜'],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜'],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜'],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜'],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜']
    ]
    features = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
    numList = [len(featureDic[feature]) for feature in features]
    newDataSet = np.array(dataSet)

    # 划分训练集和剪枝集
    trainIndex = [0, 1, 2, 5, 6, 9, 13, 14, 15, 16, 3]
    trainDataSet = newDataSet[trainIndex]
    pruneIndex = [4, 7, 8, 10, 11, 12]
    pruneDataSet = newDataSet[pruneIndex]

    return np.array(dataSet), trainDataSet, pruneDataSet, features


# 计算基尼指数
def calGini(dataArr):
    numEntries = dataArr.shape[0]
    classArr = dataArr[:, -1]
    uniqueClass = list(set(classArr))
    Gini = 1.0
    for c in uniqueClass:
        Gini -= (len(dataArr[dataArr[:, -1] == c]) / float(numEntries)) ** 2
    return Gini


# 按照某个特征值划分数据集
def splitDataSet(dataSet, ax, value):
    return np.delete(dataSet[dataSet[:, ax] == value], ax, axis=1)


# 计算分裂后的基尼指数
def calSplitGin(dataSet, ax, labels):
    newGini = 0.0
    for j in featureDic[ax]:
        axIndex = labels.index(ax)
        subDataSet = splitDataSet(dataSet, axIndex, j)
        prob = len(subDataSet) / float(len(dataSet))
        if prob != 0:
            newGini += prob * calGini(subDataSet)
    return newGini


# 选择最佳划分特征
def chooseBestSplit(dataSet, labelList):
    bestGain = 1
    bestFeature = -1
    n = dataSet.shape[1]
    for i in range(n - 1):
        newGini = calSplitGin(dataSet, labelList[i], labelList)
        if newGini < bestGain:
            bestFeature = i
            bestGain = newGini
    return bestFeature


# 计算类标签中出现最多的标签
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount:
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 递归创建决策树
def createTree(dataSet, labels):
    classList = dataSet[:, -1]
    if calGini(dataSet) == 0:
        return dataSet[0][-1]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestSplit(dataSet, labels)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    labelsCopy = labels[:]
    del (labelsCopy[bestFeat])
    uniqueVals = featureDic[bestFeatLabel]
    for value in uniqueVals:
        subLabels = labelsCopy[:]
        subDataSet = splitDataSet(dataSet, bestFeat, value)
        if len(subDataSet) != 0:
            myTree[bestFeatLabel][value] = createTree(subDataSet, subLabels)
        else:
            myTree[bestFeatLabel][value] = majorityCnt(classList)
    return myTree


# 可视化决策树
def plotTree(tree, parentPt, nodeTxt, ax):
    numLeafs = getNumLeafs(tree)
    depth = getTreeDepth(tree)
    firstStr = list(tree.keys())[0]
    secondDict = tree[firstStr]
    centerPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(centerPt, parentPt, nodeTxt, ax)
    plotNode(firstStr, centerPt, parentPt, ax)
    plotTree.yOff -= 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if isinstance(secondDict[key], dict):
            plotTree(secondDict[key], centerPt, str(key), ax)
        else:
            plotTree.xOff += 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), centerPt, ax)
            plotMidText((plotTree.xOff, plotTree.yOff), centerPt, str(key), ax)
    plotTree.yOff += 1.0 / plotTree.totalD


# 获取叶子节点数
def getNumLeafs(tree):
    numLeafs = 0
    firstStr = list(tree.keys())[0]
    secondDict = tree[firstStr]
    for key in secondDict.keys():
        if isinstance(secondDict[key], dict):
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


# 获取树的深度
def getTreeDepth(tree):
    maxDepth = 0
    firstStr = list(tree.keys())[0]
    secondDict = tree[firstStr]
    for key in secondDict.keys():
        if isinstance(secondDict[key], dict):
            thisDepth = getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


# 绘制节点的文本信息
def plotMidText(cntrPt, parentPt, txtString, ax):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    ax.text(xMid, yMid, txtString, va="center", ha="center", fontsize=12, fontproperties=font)


# 绘制节点
def plotNode(nodeTxt, cntrPt, parentPt, ax):
    ax.text(cntrPt[0], cntrPt[1], nodeTxt, va="center", ha="center", fontsize=12, fontproperties=font)
    ax.plot([parentPt[0], cntrPt[0]], [parentPt[1], cntrPt[1]], 'k-', lw=2)


# 主函数
if __name__ == "__main__":
    # 获取数据
    dataSet, trainDataSet, pruneDataSet, features = getDataSet()
    tree = createTree(trainDataSet, features)

    fig = plt.figure(figsize=(12, 8), dpi=80)
    ax = fig.add_subplot(111, frameon=False)
    plotTree.xOff = 0.0
    plotTree.yOff = 1.0
    plotTree.totalW = float(getNumLeafs(tree))
    plotTree.totalD = float(getTreeDepth(tree))
    plotTree(tree, (0.5, 1.0), '决策树', ax)
    plt.show()


# 修改递归创建决策树的函数，加入预剪枝逻辑
def createPrunedTree(dataSet, labels, min_samples_split=4, min_gini_decrease=0.1):
    classList = dataSet[:, -1]
    # 如果所有样本属于同一类，则直接返回该类别
    if calGini(dataSet) == 0:
        return dataSet[0][-1]
    # 如果样本数小于最小分裂样本数，停止分裂
    if len(dataSet) < min_samples_split:
        return majorityCnt(classList)
    # 如果没有更多特征可分裂，返回类别多数票
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 选择最佳分裂特征
    bestFeat = chooseBestSplit(dataSet, labels)
    bestFeatLabel = labels[bestFeat]
    # 计算分裂前后的基尼指数差
    current_gini = calGini(dataSet)
    new_gini = calSplitGin(dataSet, labels[bestFeat], labels)
    gini_decrease = current_gini - new_gini

    # 如果基尼指数减少不显著，停止分裂
    if gini_decrease < min_gini_decrease:
        return majorityCnt(classList)
    # 构建子树
    myTree = {bestFeatLabel: {}}
    labelsCopy = labels[:]
    del (labelsCopy[bestFeat])
    uniqueVals = featureDic[bestFeatLabel]
    for value in uniqueVals:
        subLabels = labelsCopy[:]
        subDataSet = splitDataSet(dataSet, bestFeat, value)
        # 如果子数据集为空，使用多数票填充
        if len(subDataSet) == 0:
            myTree[bestFeatLabel][value] = majorityCnt(classList)
        else:
            myTree[bestFeatLabel][value] = createPrunedTree(subDataSet, subLabels, min_samples_split, min_gini_decrease)
    return myTree


# 主程序
if __name__ == "__main__":
    # 获取数据
    dataSet, trainDataSet, pruneDataSet, features = getDataSet()
    # 创建预剪枝决策树
    pruned_tree = createPrunedTree(trainDataSet, features, min_samples_split=4, min_gini_decrease=0.1)

    # 绘制预剪枝后的决策树
    fig = plt.figure(figsize=(12, 8), dpi=80)
    ax = fig.add_subplot(111, frameon=False)
    plotTree.xOff = 0.0
    plotTree.yOff = 1.0
    plotTree.totalW = float(getNumLeafs(pruned_tree))
    plotTree.totalD = float(getTreeDepth(pruned_tree))
    plotTree(pruned_tree, (0.5, 1.0), '预剪枝决策树', ax)
    plt.show()

