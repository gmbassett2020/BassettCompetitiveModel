# Develop simulated football season.
# Connect to a Java Neural Network using Lasagne.
# http://lasagne.readthedocs.org/
#    More in-depth examples and reproductions of paper results are maintained in
#    a separate repository: https://github.com/Lasagne/Recipes
# 2022-01-17
# 2022-08-05
#  Moved from jupyter-notebook to python file.  
#  Removing neural net items since focusing first on purely linear model.

import math
from scipy import special

import random
   # Syntax : random.gauss(mu, sigma)
   #
   # Parameters :
   # mu : mean
   # sigma : standard deviation
   #
   # Returns : a random gaussian distribution floating number
   #
   # If we're trying to establish equivalency between RMS and standard deviation, .... if the mean is zero, as is often 
   # the case in electrical signals, there is no difference between the RMS calculation and the 
   # standard-deviation calculation.Jul 28, 2020

import sys

import pandas

# Season classes

import team
import game
import gameSeason
import gameSimulator

#
# Settings
#

# Default values

seasonRandomSeed = 20210905
fitRandomSeed = 20210905
numberOfSeasons = 250
learningRateInitial = 0.00001
maxLearningRate = 0.04
numberOfPassesThroughAllSeasons = 500
numberOfIterationsForEachSeason = 1
verbose = False

optimizeWeekWeights = False
randomizeWeights = True
randomizeTeamPowers = True

spreadRms = 14
rmsMultiplier = 1.0
roundScores = True

useTeamListFile = None
teamList = None
useTeamSeasonsFile = None
teamSeasons = None
useTeamPowersFile = None
saveTeamPowersToFile = None

adjustPowerAndLinearParametersFromScores = True
correlatePreviousSeasonsPower = None # default: not adjustPowerAndLinearParametersFromScores
   # Set to False to use a completely random power for previous year/first guess for current year.
   # I.e. actual power, used to generate scores, is not correlated with initial power guess.  
   # Set to False with adjustPowerAndLinearParametersFromScores.
randomizeLinearParameters = False
fitTeamPowers = None # default: not correlatePreviousSeasonsPower
minimizeChangePowerPerSeason = None # default: fitTeamPowers and correlatePreviousSeasonsPower
splitFitIntoComponents = False
fitHomeFieldAdvantage = False
costChangeTargetPercentage = 1
costChangeMaxPercentage = 10
costChangeMaximumPercentage = 10
costChangeStopPercentage = 0.001
costChangePercentAtCostValueTargets = "0,0.1"
costChangePercentAndMaxAtCostValueTargets = ""
deprioritizeTotalForCost = 0.25
constrainAveragePower = True
forceAverageOffense = None
forceAverageDefense = None

#2023-07-28
# found linear parameters with this (averaged over the 5 cases):
#simFb-s20230727-f4-forceAve-adjustLin.out
#spdCoefOff	spdCoefDef	totCoefOff	totCoefDef	totCoef0
#7.45	4.7	2.83	-2.95	24.26

spdCoefOffInit = 7.45
spdCoefDefInit = 4.7
totCoef0Init = 24.26
totCoefOffInit = 2.83
totCoefDefInit = -2.95

adamUpdateEnable = False # use Adam - a method for stochasitc optimization.  https://arxiv.org/pdf/1412.6980.pdf

print("ARGV: "+str(sys.argv))

# Overide with command line arguents
if (len(sys.argv) > 1):
   for iArg in range(1,len(sys.argv)):
      keyAndValueString = sys.argv[iArg].split("=",2)

      if (keyAndValueString[0] == "seasonRandomSeed"):
         seasonRandomSeed = int(keyAndValueString[1])
         print("Setting seasonRandomSeed to "+str(seasonRandomSeed))

      elif (keyAndValueString[0] == "fitRandomSeed"):
         fitRandomSeed = int(keyAndValueString[1])
         print("Setting fitRandomSeed to "+str(fitRandomSeed))

      elif (keyAndValueString[0] == "numberOfSeasons"):
         numberOfSeasons = int(keyAndValueString[1])
         print("Setting numberOfSeasons to "+str(numberOfSeasons))
   
      elif (keyAndValueString[0] == "learningRateInitial"):
         learningRateInitial = float(keyAndValueString[1])
         print("Setting learningRateInitial to "+str(learningRateInitial))

      elif (keyAndValueString[0] == "maxLearningRate"):
         maxLearningRate = float(keyAndValueString[1])
         print("Setting maxLearningRate to "+str(maxLearningRate))

      elif (keyAndValueString[0] == "costChangeStopPercentage"):
         costChangeStopPercentage = float(keyAndValueString[1])
         print("Setting costChangeStopPercentage to "+str(costChangeStopPercentage))

      elif (keyAndValueString[0] == "costChangeTargetPercentage"):
         costChangeTargetPercentage = float(keyAndValueString[1])
         print("Setting costChangeTargetPercentage to "+str(costChangeTargetPercentage))

      elif (keyAndValueString[0] == "costChangePercentAtCostValueTargets"):
         costChangePercentAtCostValueTargets = keyAndValueString[1]
         print("Setting costChangePercentAtCostValueTargets to "+str(costChangePercentAtCostValueTargets))

      elif (keyAndValueString[0] == "costChangePercentAndMaxAtCostValueTargets"):
         costChangePercentAndMaxAtCostValueTargets = keyAndValueString[1]
         print("Setting costChangePercentAndMaxAtCostValueTargets to "+str(costChangePercentAndMaxAtCostValueTargets))

      elif (keyAndValueString[0] == "numberOfPassesThroughAllSeasons"):
         numberOfPassesThroughAllSeasons = int(keyAndValueString[1])
         print("Setting numberOfPassesThroughAllSeasons to "+str(numberOfPassesThroughAllSeasons))

      elif (keyAndValueString[0] == "numberOfIterationsForEachSeason"):
         numberOfIterationsForEachSeason = int(keyAndValueString[1])
         print("Setting numberOfIterationsForEachSeason to "+str(numberOfIterationsForEachSeason))

      elif (keyAndValueString[0] == "deprioritizeTotalForCost"):
         deprioritizeTotalForCost = float(keyAndValueString[1])
         print("Setting deprioritizeTotalForCost to "+str(deprioritizeTotalForCost))

      elif (keyAndValueString[0] == "optimizeWeekWeights"):
         optimizeWeekWeights = (keyAndValueString[1] == "True" or keyAndValueString[1] == "true")
         print("Setting optimizeWeekWeights to "+str(optimizeWeekWeights))

      elif (keyAndValueString[0] == "randomizeWeights"):
         randomizeWeights = (keyAndValueString[1] == "True" or keyAndValueString[1] == "true")
         print("Setting randomizeWeights to "+str(randomizeWeights))

      elif (keyAndValueString[0] == "randomizeTeamPowers"):
         randomizeTeamPowers = (keyAndValueString[1] == "True" or keyAndValueString[1] == "true")
         print("Setting randomizeTeamPowers to "+str(randomizeTeamPowers))

      elif (keyAndValueString[0] == "adjustPowerAndLinearParametersFromScores"):
         adjustPowerAndLinearParametersFromScores = (keyAndValueString[1] == "True" or keyAndValueString[1] == "true")
         print("Setting adjustPowerAndLinearParametersFromScores to "+str(adjustPowerAndLinearParametersFromScores))

      elif (keyAndValueString[0] == "correlatePreviousSeasonsPower"):
         correlatePreviousSeasonsPower = (keyAndValueString[1] == "True" or keyAndValueString[1] == "true")
         print("Setting correlatePreviousSeasonsPower to "+str(correlatePreviousSeasonsPower))

      elif (keyAndValueString[0] == "randomizeLinearParameters"):
         randomizeLinearParameters = (keyAndValueString[1] == "True" or keyAndValueString[1] == "true")
         print("Setting randomizeLinearParameters to "+str(randomizeLinearParameters))

      elif (keyAndValueString[0] == "fitTeamPowers"):
         fitTeamPowers = (keyAndValueString[1] == "True" or keyAndValueString[1] == "true")
         print("Setting fitTeamPowers to "+str(fitTeamPowers))

      elif (keyAndValueString[0] == "fitLinearParameters"):
         fitLinearParameters = (keyAndValueString[1] == "True" or keyAndValueString[1] == "true")
         print("Setting fitLinearParameters to "+str(fitLinearParameters))

      elif (keyAndValueString[0] == "fitHomeFieldAdvantage"):
         fitHomeFieldAdvantage = (keyAndValueString[1] == "True" or keyAndValueString[1] == "true")
         print("Setting fitHomeFieldAdvantage to "+str(fitHomeFieldAdvantage))

      elif (keyAndValueString[0] == "minimizeChangePowerPerSeason"):
         minimizeChangePowerPerSeason = (keyAndValueString[1] == "True" or keyAndValueString[1] == "true")
         print("Setting minimizeChangePowerPerSeason to "+str(minimizeChangePowerPerSeason))

      elif (keyAndValueString[0] == "roundScores"):
         roundScores = (keyAndValueString[1] == "True" or keyAndValueString[1] == "true")
         print("Setting roundScores to "+str(roundScores))

      elif (keyAndValueString[0] == "verbose"):
         verbose = (keyAndValueString[1] == "True" or keyAndValueString[1] == "true")
         print("Setting verbose to "+str(verbose))

      elif (keyAndValueString[0] == "splitFitIntoComponents"):
         splitFitIntoComponents = (keyAndValueString[1] == "True" or keyAndValueString[1] == "true")
         print("Setting splitFitIntoComponents to "+str(splitFitIntoComponents))

      elif (keyAndValueString[0] == "adamUpdateEnable"):
         adamUpdateEnable = (keyAndValueString[1] == "True" or keyAndValueString[1] == "true")
         print("Setting adamUpdateEnable to "+str(adamUpdateEnable))

      elif (keyAndValueString[0] == "rmsMultiplier"):
         rmsMultiplier = float(keyAndValueString[1])
         print("Setting rmsMultiplier to "+str(rmsMultiplier))

      elif (keyAndValueString[0] == "constrainAveragePower"):
         constrainAveragePower = (keyAndValueString[1] == "True" or keyAndValueString[1] == "true")
         print("Setting constrainAveragePower to "+str(constrainAveragePower))

      elif (keyAndValueString[0] == "forceAverageOffense"):
         forceAverageOffense = float(keyAndValueString[1])
         print("Setting forceAverageOffense to "+str(forceAverageOffense))

      elif (keyAndValueString[0] == "forceAverageDefense"):
         forceAverageDefense = float(keyAndValueString[1])
         print("Setting forceAverageDefense to "+str(forceAverageDefense))

      elif (keyAndValueString[0] == "useTeamListFile"):
         useTeamListFile = keyAndValueString[1]
         print("Loading team info from "+useTeamListFile)

      elif (keyAndValueString[0] == "useTeamSeasonsFile"):
         useTeamSeasonsFile = keyAndValueString[1]
         print("Loading season info from "+useTeamSeasonsFile)

      elif (keyAndValueString[0] == "useTeamPowersFile"):
         useTeamPowersFile = keyAndValueString[1]
         print("Saving team powers to "+useTeamPowersFile)

      elif (keyAndValueString[0] == "saveTeamPowersToFile"):
         saveTeamPowersToFile = keyAndValueString[1]
         print("Saving team powers to "+saveTeamPowersToFile)

      elif (keyAndValueString[0] == "spdCoefOffInit"):
         spdCoefOffInit = float(keyAndValueString[1])
         print("Setting spdCoefOffInit to "+str(spdCoefOffInit))

      elif (keyAndValueString[0] == "spdCoefDefInit"):
         spdCoefDefInit = float(keyAndValueString[1])
         print("Setting spdCoefDefInit to "+str(spdCoefDefInit))

      elif (keyAndValueString[0] == "totCoefOffInit"):
         totCoefOffInit = float(keyAndValueString[1])
         print("Setting totCoefOffInit to "+str(totCoefOffInit))

      elif (keyAndValueString[0] == "totCoefDefInit"):
         totCoefDefInit = float(keyAndValueString[1])
         print("Setting totCoefDefInit to "+str(totCoefDefInit))

      elif (keyAndValueString[0] == "totCoef0Init"):
         totCoef0Init = float(keyAndValueString[1])
         print("Setting totCoef0Init to "+str(totCoef0Init))

      else:
         print("ERROR: unknown parameter "+keyAndValueString[0])
         raise Exception("ERROR: unknown parameter")

random.seed(seasonRandomSeed)

costChangeArray = []
if (costChangePercentAndMaxAtCostValueTargets != ""):
   costChangeStrings = costChangePercentAndMaxAtCostValueTargets.split(',')
   for changeString in costChangeStrings:
      costChangeArray.append(float(changeString))
else:
   costChangeStrings = costChangePercentAtCostValueTargets.split(',')
   count = 0
   for changeString in costChangeStrings:
      costChangeArray.append(float(changeString))
      count += 1
      if (count%2 == 0):
         costChangeArray.append(10.)

if (correlatePreviousSeasonsPower == None):
   correlatePreviousSeasonsPower = not adjustPowerAndLinearParametersFromScores
   # Set to False to use a completely random power for previous year/first guess for current year.
   # I.e. actual power, used to generate scores, is not correlated with initial power guess.  
   # Set to False with adjustPowerAndLinearParametersFromScores.

if (fitTeamPowers == None):
   fitTeamPowers = not correlatePreviousSeasonsPower

if (minimizeChangePowerPerSeason == None):
   minimizeChangePowerPerSeason = fitTeamPowers and correlatePreviousSeasonsPower

#
# Set initial values for linear parameters
#

homeAdvantageInit = 3.
#first set
#spdCoefOffInit = 18.4
#spdCoefDefInit = 17.6 
#totCoef0Init = -17 
#totCoefOffInit = 14.8 
#totCoefDefInit = -9.6 
#rmsCoef0Init = rmsMultiplier * 16.173 
#rmsCoefOffInit = rmsMultiplier * 1.324 
#rmsCoefDefInit = rmsMultiplier * -1.684
# wip
#spdCoefOffInit = 18.4
#spdCoefDefInit = 25.8
#totCoef0Init = 0
#totCoefOffInit = 14.8 
#totCoefDefInit = -9.6 
#rmsCoef0Init = rmsMultiplier * 16.173 
#rmsCoefOffInit = rmsMultiplier * 1.324 
#rmsCoefDefInit = rmsMultiplier * -1.684
maxConferenceGameWeights = 10

gameSimulator = gameSimulator.GameSimulator(homeAdvantageInit, spdCoefOffInit, spdCoefDefInit, totCoef0Init, totCoefOffInit, totCoefDefInit, roundScores, spreadRms*rmsMultiplier, maxConferenceGameWeights)


#
# Set up the seasons
#

gameSeasons = []

if (useTeamListFile != None and useTeamSeasonsFile != None):
   teamList = pandas.read_csv(useTeamListFile)
   teamSeasons = pandas.read_csv(useTeamSeasonsFile)

   gameSeasonDict = {}
   roundsByYearDict = {}

   # Add all teams and seasons from the list file.
   for listIndex in teamList.index:
      if (roundsByYearDict.get(teamList["Year"][listIndex]) == None):
         roundsByYearDict[teamList["Year"][listIndex]] = {}
      if (gameSeasonDict.get(teamList["Year"][listIndex]) == None):
         print("Creating gameSeason for "+str(teamList["Year"][listIndex]))
         thisGameSeason = gameSeason.GameSeason(teamList["Year"][listIndex], gameSimulator, "gs")
         gameSeasonDict[teamList["Year"][listIndex]] = thisGameSeason
         print("Created gameSeason "+str(teamList["Year"][listIndex])+" with seasonYear "+str(thisGameSeason.seasonYear))
      teamObject = team.Team.teamObjectByName.get(teamList["Team"][listIndex])
      if (teamObject == None):
         teamObject = team.Team(teamList["Team"][listIndex])
      teamObject.SetConferenceAndDivision(teamList["Year"][listIndex], teamList["Conference"][listIndex], teamList["Division"][listIndex])

   for gameIndex in teamSeasons.index:
      thisGameSeason = gameSeasonDict.get(teamSeasons["Season"][gameIndex])
      team1Object = team.Team.teamObjectByName.get(teamSeasons["Team1"][gameIndex])
      team2Object = team.Team.teamObjectByName.get(teamSeasons["Team2"][gameIndex])
      year = teamSeasons["Season"][gameIndex]
      roundName = teamSeasons["RoundName"][gameIndex]
      roundNumber = teamSeasons["RoundNumber"][gameIndex]
      homeField = 0.5
      gameRowString = "Year="+str(year)+" team1="+teamSeasons["Team1"][gameIndex]+" team2="+teamSeasons["Team2"][gameIndex]+" roundName="+roundName+" roundNumber="+str(roundNumber)+" homeFlag="+teamSeasons["HomeFlag"][gameIndex]+" score1="+str(teamSeasons["Score1"][gameIndex])+" score2="+str(teamSeasons["Score2"][gameIndex])
      if (teamSeasons["HomeFlag"][gameIndex] == "team1Home"):
         homeField = 1.0
      elif (teamSeasons["HomeFlag"][gameIndex] == "team2Home"):
         homeField = 0.0
      if (thisGameSeason == None):
         print("ERROR: unknown year in gameRow "+gameRowString)
         exit(1)
      if (team1Object == None):
         print("WARNING: skipping game because of unknown Team1="+str(teamSeasons["Team1"][gameIndex])+" in gameRow "+gameRowString)
      if (team2Object == None):
         print("WARNING: skipping game because of unknown Team2="+str(teamSeasons["Team2"][gameIndex])+" in gameRow "+gameRowString)
      if (team1Object != None and team2Object != None and team1Object.GetConferenceName(year) != None and team2Object.GetConferenceName(year) != None):
         # Only add a game if both teams are active for that season, determined by if they have a conference defined.
         if (roundsByYearDict[year].get(roundNumber) == None):
            # Add new round.  
            roundsByYearDict[year][roundNumber] = roundName
            #print("::: adding roundName "+roundName+" year "+str(year)+" thisGameSeason.seasonYear "+str(thisGameSeason.seasonYear))
            thisGameSeason.roundNames.append(roundName)
            thisGameSeason.seasonInfo[roundName] = [] # array containing Game objects
            thisGameSeason.roundIndex[roundName] = roundNumber
            thisGameSeason.roundCount += 1
         print("::: adding game for year "+str(thisGameSeason.seasonYear)+" team1 "+team1Object.teamName+" conf "+team1Object.GetConferenceName(year)+" team2 "+team2Object.teamName+" conf "+team2Object.GetConferenceName(year)+" roundName "+roundName+" s1 "+str(teamSeasons["Score1"][gameIndex])+" s2 "+str(teamSeasons["Score2"][gameIndex]))
         thisGameSeason.AddGame(team1Object, team2Object, roundName, homeField, False, teamSeasons["Score1"][gameIndex], teamSeasons["Score2"][gameIndex])
         team1Object.SetOpponent(year, roundName, team2Object, homeField)
         team2Object.SetOpponent(year, roundName, team1Object, homeField-1.0)

   for keyIndex in sorted(gameSeasonDict.keys()):
      thisGameSeason = gameSeasonDict[keyIndex]
      gameSeasons.append(gameSeasonDict[keyIndex])

   if (verbose):
      print("SEASONS:")
      print("")
      for thisGameSeason in gameSeasons:
         thisGameSeason.PrintSeason()
         print("")
         print("")

else:

   year = 2022
   for count in range(numberOfSeasons):
      #thisGameSeason = GameSeason(year+count, gameSimulator, "gs"+str(year+count))
      thisGameSeason = gameSeason.GameSeason(year+count, gameSimulator, "gs")
      thisGameSeason.CreateSimulatedSeason("NCAA2Divisions", correlatePreviousSeasonsPower, optimizeWeekWeights)
      gameSeasons.append(thisGameSeason)
      print("Creating season "+str(count))
      if (verbose):
         thisGameSeason.PrintSeason()
         print("")
         print("")

print("::: useTeamPowersFile="+str(useTeamPowersFile))
if (useTeamPowersFile != None):
   teamPowers = pandas.read_csv(useTeamPowersFile)
   # ,teamName,teamId,year,offense,defense,offenseRms,defenseRms
   for powersIndex in teamPowers.index:
      teamObject = team.Team.teamObjectById[teamPowers["teamId"][powersIndex]]
      teamObject.SetPower(teamPowers["year"][powersIndex], teamPowers["offense"][powersIndex], teamPowers["defense"][powersIndex])
      teamObject.SetPowerActual(teamPowers["year"][powersIndex], teamPowers["offense"][powersIndex], teamPowers["defense"][powersIndex])

# Now that all the scores have been generated, reset the offensive & defensive powers
if (randomizeTeamPowers):
   for thisGameSeason in gameSeasons:
      thisGameSeason.RandomizeTeamPower(5, 15, 0, 10)

#
# Test out fitting for one season
#

random.seed(fitRandomSeed)

if (optimizeWeekWeights):
   print("")
   print("Optimizing week weights")
   if (randomizeWeights):
      gameSimulator.RandomizeWeights()
   else:
      gameSimulator.SetWeights()
   print("Initial weights: "+str(gameSimulator.weights))
else:
   gameSimulator.SetWeightsToBestFit()
   print("Week weights: "+str(gameSimulator.weights))

if (randomizeLinearParameters or fitHomeFieldAdvantage):
   print("Randomizing linear parameters.")
   gameSimulator.RandomizeLinearParameters(fitHomeFieldAdvantage, randomizeLinearParameters)

if (not correlatePreviousSeasonsPower):
   print("First guess on power is not correlated with actual power.")

elementsToFit = []
if (splitFitIntoComponents):
   # Run through each fit item one at a time.
   print("ERROR: logic for splitFitIntoComponents needs work before enabling")
   exit(1)
   if (fitTeamPowers):
      elementsToFit.append({'teamPowers': True, 'homeFieldAdvantage': False, 'deprioritizeTotalForCost': deprioritizeTotalForCost})
   if (fitHomeFieldAdvantage):
      elementsToFit.append({'teamPowers': False, 'homeFieldAdvantage': True, 'deprioritizeTotalForCost': deprioritizeTotalForCost})
else:
   elementsToFit.append({'teamPowers': fitTeamPowers, 'linearParameters': fitLinearParameters, 'homeFieldAdvantage': fitHomeFieldAdvantage, 'deprioritizeTotalForCost': deprioritizeTotalForCost})

costChangeArrayIndex = 0
nextCostTarget = 0
if (len(costChangeArray) > costChangeArrayIndex):
   nextCostTarget = costChangeArray[costChangeArrayIndex]
   nextTargetChangePercentage = costChangeArray[costChangeArrayIndex+1]
   nextMaxChangePercentage = costChangeArray[costChangeArrayIndex+2]
   costChangeArrayIndex += 3

# compute the average team power over all seasons
averageTeamPower = 0
seasonCount = 0
for thisGameSeason in gameSeasons:
   seasonCount += 1
   averageTeamPower += thisGameSeason.averageTeamPower
averageTeamPower /= seasonCount
if (constrainAveragePower):
   print("AVERAGE TEAM POWER: will constrain average team power to be "+str(averageTeamPower))

learningRate = {}
for thisGameSeason in gameSeasons:
   learningRate[thisGameSeason.seasonYear] = learningRateInitial

for iPass in range(numberOfPassesThroughAllSeasons):

   print("iPass "+str(iPass))

   #seasonList = []
   #for thisGameSeason in gameSeasons
   #   seasonList.append(thisGameSeason)
   #random.shuffle(seasonList)
   random.shuffle(gameSeasons)

   learningRateUsedThisIteration = 1
   for thisGameSeason in gameSeasons:
      if (learningRate[thisGameSeason.seasonYear] < learningRateUsedThisIteration):
         learningRateUsedThisIteration = learningRate[thisGameSeason.seasonYear]

   for thisGameSeason in gameSeasons:

      if (optimizeWeekWeights):
         for iter in range(numberOfIterationsForEachSeason):
            print("Pass "+str(iPass)+" for season "+str(thisGameSeason.seasonYear)+" iteration "+str(iter))
            thisGameSeason.currentRound = 0
            gameSimulator.ResetDeltaWeights()
            for count in range(15):
               thisGameSeason.FitPowerForCurrentRound(learningRate[thisGameSeason.seasonYear])
               thisGameSeason.currentRound += 1
            print("Deltas: "+str(gameSimulator.deltaWeights))
            gameSimulator.ApplyDeltaWeights()
            print("Updated weights: "+str(gameSimulator.weights))
            print("")

      #thisGameSeason.RandomizeTeamPower(5, 15, 0, 10)
      #gameSimulator.RandomizeLinearParameters
      
      if (adjustPowerAndLinearParametersFromScores):
         for iter in range(numberOfIterationsForEachSeason):
            print("Pass "+str(iPass)+" for "+thisGameSeason.name+" season "+str(thisGameSeason.seasonYear)+" iteration "+str(iter))
            if (splitFitIntoComponents):
               random.shuffle(elementsToFit)
            fitVerbose = False
            if ((iPass % (numberOfPassesThroughAllSeasons/5) == (numberOfPassesThroughAllSeasons/5-1)) and iter == numberOfIterationsForEachSeason-1):
               fitVerbose = True
            for iFit in range(len(elementsToFit)):
               costInfo = thisGameSeason.AdjustPowerAndLinearParametersFromScores(
                  learningRate[thisGameSeason.seasonYear], 
                  adamUpdateEnable,
                  elementsToFit[iFit],
                  averageTeamPower,
                  0.001,
                  fitVerbose,
                  constrainAveragePower,
                  forceAverageOffense,
                  forceAverageDefense)
               # Adjust learning rate
               costReductionPercentage = 100*(costInfo["costOrig"]-costInfo["costUpdated"])/costInfo["costOrig"]
               print("::: costInfo for "+str(thisGameSeason.seasonYear)+" cost="+str(costInfo["costUpdated"])+" costReduction="+str(costReductionPercentage)+"% targetChange="+str(costChangeTargetPercentage)+"% maxCostChangePercentage="+str(costChangeMaxPercentage)+"% learningRate="+str(learningRate[thisGameSeason.seasonYear])+" nextCostTarget="+str(nextCostTarget))
               if (costInfo["costUpdated"] <= nextCostTarget):
                  nextCostTarget = 0
                  costChangeTargetPercentage = nextTargetChangePercentage
                  costChangeMaxPercentage = nextMaxChangePercentage
                  if (len(costChangeArray) > costChangeArrayIndex):
                     nextCostTarget = costChangeArray[costChangeArrayIndex]
                     nextTargetChangePercentage = costChangeArray[costChangeArrayIndex+1]
                     nextMaxChangePercentage = costChangeArray[costChangeArrayIndex+2]
                     costChangeArrayIndex += 3
               if (costChangeTargetPercentage <= costChangeStopPercentage):
                  print("Cost target percentage, "+str(costChangeTargetPercentage)+", at or below stopping threshold, "+str(costChangeStopPercentage)+".  Stopping execution.")
                  exit(0)
               if (not adamUpdateEnable):
                  if (costReductionPercentage < 0):
                     learningRate[thisGameSeason.seasonYear] /= 1.05*1.05*1.05
                     print("Cost has increased.  Reducing learning rate to "+str(learningRate)+" and target percentage to "+str(costChangeTargetPercentage)+".")
                     #print("Cost has increased.  Stopping execution.")
                     #exit(0)
                  elif (costReductionPercentage < costChangeTargetPercentage):
                     learningRate[thisGameSeason.seasonYear] *= 1.05
                     print("Cost reduction percentage, "+str(costReductionPercentage)+", below target of "+str(costChangeTargetPercentage)+".  Increasing learning rate to "+str(learningRate[thisGameSeason.seasonYear]))
                  elif (costReductionPercentage > costChangeMaxPercentage):
                     learningRate[thisGameSeason.seasonYear] /= 1.05
                     print("Cost reduction percentage, "+str(costReductionPercentage)+", above max target of "+str(costChangeMaxPercentage)+".  Decreasing learning rate to "+str(learningRate[thisGameSeason.seasonYear]))
                  elif (costReductionPercentage > costChangeMaximumPercentage):
                     learningRate[thisGameSeason.seasonYear] /= 1.05
                     print("Cost reduction percentage, "+str(costReductionPercentage)+", above max of "+str(costChangeMaximumPercentage)+".  Decreasing learning rate to "+str(learningRate[thisGameSeason.seasonYear])+" cost="+str())
                  print("::: learningRate="+str(learningRate[thisGameSeason.seasonYear])+" currentPercentage="+str(costReductionPercentage)+" target="+str(costChangeTargetPercentage))
                  if (learningRate[thisGameSeason.seasonYear] > maxLearningRate):
                     learningRate[thisGameSeason.seasonYear] = maxLearningRate
                     print("::: capping learning rate at "+str(maxLearningRate))

   # Minimize the change in power between each season.
   #
   # What is the proper factor between offense and defense powers?
   # Total score = self.totCoefOff*(o2+o1) + self.totCoefDef*(d2+d1) + self.totCoef0 
   # Change in total score for o1 & d1 = 0
   #    0 = self.totCoefOff*do1 + self.totCoefDef*dd1
   #    dd1 = -self.totCoefOff/self.totCoefDef * do1
   # 
   # team power = offensePower + gamma*defensePower
   #
   # Change cost for a team's power
   # cost_off = sum over seasons: (off_i - off_ip1)^2 
   # cost_def = sum over seasons: (def_i - def_ip1)^2
   #
   # costChangePower = cost_off + gamma^2*cost_def
   #
   # d cost_off / d off_i = -2*(off_im1-off_i) + 2*(off_i-off_ip1)
   # d cost_def / d def_i = -2*(def_im1-def_i) + 2*(def_i-def_ip1)
   #
   costChangePower = 0.0
   costChangePowerCount = 0
   gamma = -gameSimulator.totCoefOff/gameSimulator.totCoefDef
   changePowerCostFactor = 0.001
   offenseRMS = 0.0
   defenseRMS = 0.0
   countRMS = 0
   for teamName in team.Team.teamObjectByName:
      #print("::: working on team "+teamName)
      teamObject = team.Team.teamObjectByName[teamName]
      seasonYears = []
      deltaDefense = []
      deltaOffense = []
      defense = []
      offense = []
      defenseActual = []
      offenseActual = []
      thisCostChangePower = 0
      for thisGameSeason in gameSeasons:
         power = teamObject.GetPower(thisGameSeason.seasonYear)
         powerActual = teamObject.GetPowerActual(thisGameSeason.seasonYear)
         if (powerActual["offense"] == None):
            powerActual = power
         #print("::: season "+str(thisGameSeason.seasonYear)+" power "+str(power))
         if (power["defense"] != None):
            seasonYears.append(thisGameSeason.seasonYear)
            deltaDefense.append(0.0)
            deltaOffense.append(0.0)
            defense.append(power["defense"])
            offense.append(power["offense"])
            defenseActual.append(powerActual["defense"])
            offenseActual.append(powerActual["offense"])
            #print("::: adding def")
      #print("::: number of season "+str(len(seasonYears)))
      if (len(seasonYears) > 1):
         for iSeason in range(len(seasonYears)):
            if (iSeason < len(seasonYears)-1):
               thisCostChangePower += (offense[iSeason]-offense[iSeason+1])**2 + gamma*gamma*(defense[iSeason]-defense[iSeason+1])**2
            if (iSeason > 0):
               deltaOffense[iSeason] -= 2*(offense[iSeason-1]-offense[iSeason])
               deltaDefense[iSeason] -= 2*(defense[iSeason-1]-defense[iSeason])
            if (iSeason < len(seasonYears)-1):
               deltaOffense[iSeason] += 2*(offense[iSeason]-offense[iSeason+1])
               deltaDefense[iSeason] += 2*(defense[iSeason]-defense[iSeason+1])
            if (powerActual["offense"] != None):
               countRMS += 1
               offenseRMS += (powerActual["offense"] - power["offense"])**2
               defenseRMS += (powerActual["defense"] - power["defense"])**2
         costChangePower += thisCostChangePower
         costChangePowerCount += len(seasonYears)
         if (minimizeChangePowerPerSeason):
            for iSeason in range(len(seasonYears)):
               defense[iSeason] -= changePowerCostFactor * (learningRateUsedThisIteration * deltaDefense[iSeason])
               offense[iSeason] -= changePowerCostFactor * (learningRateUsedThisIteration * deltaOffense[iSeason])
               teamObject.SetPower(seasonYears[iSeason], offense[iSeason], defense[iSeason])
   costChangePower /= costChangePowerCount
   print("::: costChangePower="+str(costChangePower))
   if (countRMS > 0):
      offenseRMS = math.sqrt(offenseRMS/countRMS)
      defenseRMS = math.sqrt(defenseRMS/countRMS)

   # Compute the latest score cost. RMS for off & def done in loop above.
   countCurrent = 0
   costCurrent = 0
   for thisGameSeason in gameSeasons:
      for iRound in range(len(thisGameSeason.roundNames)):
         if (thisGameSeason.seasonInfo.get(thisGameSeason.roundNames[iRound]) != None):
            for game in thisGameSeason.seasonInfo[thisGameSeason.roundNames[iRound]]:
               power1 = game.team1Object.GetPower(thisGameSeason.seasonYear)
               power2 = game.team2Object.GetPower(thisGameSeason.seasonYear)
               scores = gameSimulator.GetScores(
                        power1["offense"], power1["defense"], power2["offense"], power2["defense"], game.homeField, False)
               countCurrent += 1
               if (deprioritizeTotalForCost == 0):
                  costCurrent += (scores["score1"] - game.score1)**2 + (scores["score2"] - game.score2)**2
               else:
                  costCurrent += ((scores["score1"] - scores["score2"] - game.score1 + game.score2)**2
                        + deprioritizeTotalForCost*(scores["score1"] + scores["score2"] - game.score1 - game.score2)**2)
   print("cost & off/def RMS for iPass="+str(iPass)+" cost="+str(costCurrent/countCurrent)+" offenseRMS="+str(offenseRMS)+" defenseRMS="+str(defenseRMS))


if (fitLinearParameters or fitHomeFieldAdvantage):
   print("::: final"
      +" spdCoefOff="+str(gameSimulator.spdCoefOff)
      +" spdCoefDef="+str(gameSimulator.spdCoefDef)
      +" totCoefOff="+str(gameSimulator.totCoefOff)
      +" totCoefDef="+str(gameSimulator.totCoefDef)
      +" totCoef0="+str(gameSimulator.totCoef0)
      +" homeFieldAdvantage="+str(gameSimulator.homeFieldAdvantage))

if (saveTeamPowersToFile != None):
   teamNames = []
   teamIds = []
   year = []
   offense = []
   defense = []
   for teamName in team.Team.teamObjectByName:
      teamObject = team.Team.teamObjectByName[teamName]
      for thisGameSeason in gameSeasons:
         power = teamObject.GetPower(thisGameSeason.seasonYear)
         if (power["defense"] != None):
            teamNames.append(teamName)
            teamIds.append(teamObject.teamId)
            year.append(thisGameSeason.seasonYear)
            defense.append(power["defense"])
            offense.append(power["offense"])
   teamPowers = {'teamName': teamNames, 'teamId': teamIds, 'year': year, 'offense': offense, 'defense': defense}
   teamPowersPandas = pandas.DataFrame(teamPowers)
   teamPowersPandas.to_csv(saveTeamPowersToFile)

