# Develop simulated football season.
# Connect to a Java Neural Network using Lasagne.
# http://lasagne.readthedocs.org/
#    More in-depth examples and reproductions of paper results are maintained in
#    a separate repository: https://github.com/Lasagne/Recipes
# 2022-01-17
# 2022-08-05
#  Moved from jupyter-notebook to python file.  
#  Removing neural net items since focusing first on purely linear model.
# 2023-08-08
#  Run one season, either simulated or from actual data.

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
#learningRateInitial = 0.001
learningRateInitial = 0.01
#numberOfSeasons = 2019 + 1 - 1998 # actual number of seasons, including the first season used only for previous season power.
#numberOfSeasons = 2
numberOfIterationsForEachWeek = 300
numberOfIterationsForWeekWeights = 300
numberOfSeasons = 3
#numberOfIterationsForEachWeek = 30
#numberOfIterationsForWeekWeights = 10

verbose = False

adamUpdateEnable = True

spreadRms = 14
rmsMultiplier = 1.0
roundScores = False

#currentSeason = 1998
currentSeason = 2020

#useTeamListFile = "colI-list-1998_2019.csv"
#useTeamSeasonsFile = "colI-seasons-1998_2019.csv"
useTeamListFile = "colI-list-2020_2023.csv"
useTeamSeasonsFile = "colI-seasons-2020_2023.csv"
#useTeamListFile = None
#useTeamSeasonsFile = None
teamList = None
teamSeasons = None
saveTeamPowersToFile = None

fitTeamPowers = True
optimizeWeekWeights = False
randomizeTeamPowers = False # This is set up to work for not randomizing team powers
initializePowerFromPrevious = not randomizeTeamPowers
trackLinearPredictions = not optimizeWeekWeights

deprioritizeTotalForCost = 0.25

forceAverageOffense = None
forceAverageDefense = None

#2023-08-08
# found linear parameters with this (averaged over the 5 cases):
# spdCoefOffInit=8.08 spdCoefDefInit=5.93 totCoefOffInit=2.76 totCoefDefInit=-3.28 totCoef0Init=28.89 

spdCoefOffInit = 8.08
spdCoefDefInit = 5.93
totCoefOffInit = 2.76
totCoefDefInit = -3.28
totCoef0Init = 28.89 

adamUpdateEnable = True # use Adam - a method for stochasitc optimization.  https://arxiv.org/pdf/1412.6980.pdf

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

      elif (keyAndValueString[0] == "learningRateInitial"):
         learningRateInitial = float(keyAndValueString[1])
         print("Setting learningRateInitial to "+str(learningRateInitial))

      elif (keyAndValueString[0] == "numberOfSeasons"):
         numberOfSeasons = int(keyAndValueString[1])
         print("Setting numberOfSeasons to "+str(numberOfSeasons))

      elif (keyAndValueString[0] == "numberOfIterationsForWeekWeights"):
         numberOfIterationsForWeekWeights = int(keyAndValueString[1])
         print("Setting numberOfIterationsForWeekWeights to "+str(numberOfIterationsForWeekWeights))

      elif (keyAndValueString[0] == "numberOfIterationsForPreviousSeason"):
         numberOfIterationsForPreviousSeason = int(keyAndValueString[1])
         print("Setting numberOfIterationsForPreviousSeason to "+str(numberOfIterationsForPreviousSeason))

      elif (keyAndValueString[0] == "deprioritizeTotalForCost"):
         deprioritizeTotalForCost = float(keyAndValueString[1])
         print("Setting deprioritizeTotalForCost to "+str(deprioritizeTotalForCost))

      elif (keyAndValueString[0] == "optimizeWeekWeights"):
         optimizeWeekWeights = (keyAndValueString[1] == "True" or keyAndValueString[1] == "true")
         print("Setting optimizeWeekWeights to "+str(optimizeWeekWeights))

#     elif (keyAndValueString[0] == "randomizeTeamPowers"):
#        randomizeTeamPowers = (keyAndValueString[1] == "True" or keyAndValueString[1] == "true")
#        print("Setting randomizeTeamPowers to "+str(randomizeTeamPowers))

      elif (keyAndValueString[0] == "roundScores"):
         roundScores = (keyAndValueString[1] == "True" or keyAndValueString[1] == "true")
         print("Setting roundScores to "+str(roundScores))

      elif (keyAndValueString[0] == "currentSeason"):
         currentSeason = int(keyAndValueString[1])
         print("Setting currentSeason to "+str(currentSeason))

      elif (keyAndValueString[0] == "verbose"):
         verbose = (keyAndValueString[1] == "True" or keyAndValueString[1] == "true")
         print("Setting verbose to "+str(verbose))

      elif (keyAndValueString[0] == "adamUpdateEnable"):
         adamUpdateEnable = (keyAndValueString[1] == "True" or keyAndValueString[1] == "true")
         print("Setting adamUpdateEnable to "+str(adamUpdateEnable))

      elif (keyAndValueString[0] == "rmsMultiplier"):
         rmsMultiplier = float(keyAndValueString[1])
         print("Setting rmsMultiplier to "+str(rmsMultiplier))

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

#2023-08-21
#Found optimized weights using the following:
#numberOfSeasons = 100
#numberOfIterationsForEachWeek = 300
#numberOfIterationsForWeekWeights = 300
# iWeekWeightIter 299 cost [2.540729170488716, 3.2494912466686587, 2.9159332983007564, 2.492151096500357, 2.078530999979104, 1.6805707654898485, 1.3180204509279119, 0.9948134722125765, 0.7095372492187236, 0.4895885685340341, 0.31366504326693634, 0.23683016621781747, 0.18779311163478776, 0, 0, 0, 0, 0, 0, 0]
# iWeekWeightIter 299 weight [0.8140645979562876, 0.6598250349845416, 0.6760580664368508, 0.7159214887274042, 0.7564336711722774, 0.7984806907197232, 0.8401389606679754, 0.8776309277658789, 0.910508970314625, 0.9389177480378471, 0.9601690539204841, 0.9689983845809068, 0.9903137951598799, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
#2023-08-22
#New weights after removing rounding from predictions and adjusting initialization
# iWeekWeightIter 299 cost [6.061021605744927, 4.721845012974696, 3.643352469778301, 2.8629141073363282, 2.25540330871929, 1.74958077168523, 1.3309976628132625, 0.9841611102952907, 0.6954937947594652, 0.47059225292020385, 0.2938697212488857, 0.21901620500612098, 0.17622912277768846, 0, 0, 0, 0, 0, 0, 0]
# iWeekWeightIter 299 weight [0.7941167790817569, 0.7195257962111826, 0.7574704435994234, 0.7948800055001095, 0.8262671256461386, 0.8557465042369405, 0.8813065655037028, 0.9058579695882067, 0.928092095456229, 0.9493606398415331, 0.9662703181070563, 0.9720919860088111, 0.9938275247806949, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
#3190.843u 0.326s 53:11.38 99.9% 0+0k 0+14048io 0pf+0w
if (not optimizeWeekWeights):
   gameSimulator.SetWeekWeightsToBestFit()

def GetDefaultPower(divisionName):
   
   defAve = [7.417854504803192, 5.613700341866554]
   offAve = [10.264432962217683, 7.694884091447714]
   iDiv = 0
   if (divisionName == "IAA"):
      iDiv = 1

   power = {}
   power["offense"] = offAve[iDiv]
   power["defense"] = defAve[iDiv]

   return power

#
# Set up the seasons
#

gameSeasons = []
minimalFirstRound = False

if (useTeamListFile != None and useTeamSeasonsFile != None):
   teamList = pandas.read_csv(useTeamListFile)
   teamSeasons = pandas.read_csv(useTeamSeasonsFile)

   minimalFirstRound = True

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
         # Start out by initializing with average power per division.  This will be overridden to previous seasons power when team continues from previous year.
         avePower1 = GetDefaultPower(team1Object.GetDivisionName(year))
         team1Object.SetPower(year, avePower1["offense"], avePower1["defense"])
         avePower2 = GetDefaultPower(team2Object.GetDivisionName(year))
         team2Object.SetPower(year, avePower2["offense"], avePower2["defense"])
         #print("::: adding game for year "+str(thisGameSeason.seasonYear)+" team1 "+team1Object.teamName+" conf "+team1Object.GetConferenceName(year)+" team2 "+team2Object.teamName+" conf "+team2Object.GetConferenceName(year)+" roundName "+roundName+" s1 "+str(teamSeasons["Score1"][gameIndex])+" s2 "+str(teamSeasons["Score2"][gameIndex]))
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

   print("::: createSimulatedSeason")
   uniqueTeamNames = True
   for count in range(numberOfSeasons):
      thisGameSeason = gameSeason.GameSeason(currentSeason, gameSimulator, "gs"+str(count))
      thisGameSeason.CreateSimulatedSeason("NCAA2Divisions", True, initializePowerFromPrevious, uniqueTeamNames)
      gameSeasons.append(thisGameSeason)
      if (verbose):
         print("::: gameSeason "+thisGameSeason.name+" seasonYear "+str(thisGameSeason.seasonYear))
         thisGameSeason.PrintSeason()
         print("::: by schedules")
         thisGameSeason.PrintTeamSchedules()
         print("")
         print("")

#
# 
#

random.seed(fitRandomSeed)

elementsToFit = []
elementsToFit.append({'teamPowers': fitTeamPowers, 'linearParameters': False, 'homeFieldAdvantage': False, 'deprioritizeTotalForCost': deprioritizeTotalForCost})

if (fitTeamPowers):
   fitResults = []
   #for thisGameSeason in gameSeasons:
   for iSeason in range(numberOfSeasons):
      thisGameSeason = gameSeasons[iSeason]
      gameSimulator.adamParams["beta1t"] = 1.0
      gameSimulator.adamParams["beta2t"] = 1.0
      previousTotalCount = -1
      fitResults.append({})
      for iRound in range(thisGameSeason.roundCount):
         costInfo = {}
         reachedStoppingCost = False
         #iterationMultiplier = (iRound < 3) ? 3 : 0
         iterationMultiplier = 1
         for iPass in range(iterationMultiplier*numberOfIterationsForEachWeek):
            if (not reachedStoppingCost):
               #print("gameSeason "+thisGameSeason.name+" season "+str(thisGameSeason.seasonYear)+" round "+thisGameSeason.roundNames[iRound]+ " iPass "+str(iPass))
               for iFit in range(len(elementsToFit)):
                  costInfo = thisGameSeason.AdjustPowerAndLinearParametersFromScores(
                     learningRateInitial,
                     adamUpdateEnable,
                     elementsToFit[iFit],
                     None,
                     0.001,
                     verbose, # verbose
                     False,
                     forceAverageOffense,
                     forceAverageDefense,
                     iRound+1)
                  # Adjust learning rate
                  costReductionPercentage = 100*(costInfo["costOrig"]-costInfo["costUpdated"])/max(1,costInfo["costOrig"])
                  if (iPass == 0 or iPass == numberOfIterationsForEachWeek-1 or iPass % 10 == 0):
                     print("::: costInfo for "+thisGameSeason.name+" "+str(thisGameSeason.seasonYear)+" round "+thisGameSeason.roundNames[iRound]+" pass "+str(iPass)+" cost="+str(costInfo["costUpdated"])+" costReduction="+str(costReductionPercentage)+"% gameCount="+str(costInfo["count"]))
                  #if (costReductionPercentage < 0.0001 or costInfo["costUpdated"] == 0):
                  if (costInfo["costOrig"] == 0 or costInfo["count"] == previousTotalCount):
                     reachedStoppingCost = True
         previousTotalCount = costInfo["count"] 
         # Store the fit results by team.
         if (thisGameSeason.seasonInfo.get(thisGameSeason.roundNames[iRound]) != None):
            for game in thisGameSeason.seasonInfo[thisGameSeason.roundNames[iRound]]:
               # TODO: save a pointer to the game to be able to access later: gameIndex = thisGameSeason.gameIndexByTeamAndRound[thisGameSeason.roundNames[iRound]][game.team1???] FIXME
               if (fitResults[iSeason].get(game.team1Object.teamId) == None):
                  fitResults[iSeason][game.team1Object.teamId] = [] # array of games for each team
               team1Results = {}
               power1 = game.team1Object.GetPower(thisGameSeason.seasonYear)
               team1Results["roundName"] = thisGameSeason.roundNames[iRound]
               team1Results["roundIndex"] = iRound
               team1Results["defenseFit"] = power1["defense"]
               team1Results["offenseFit"] = power1["offense"]
               fitResults[iSeason][game.team1Object.teamId].append(team1Results)
               # The "actual" power will be the fit power for the entire season.  Will go ahead and fill it in
               # here, updating it for each round, leaving the final update to the the "actual"
               game.team1Object.SetPowerActual(thisGameSeason.seasonYear, power1["offense"], power1["defense"])
               game.team1Object.SetFitPower(thisGameSeason.seasonYear, thisGameSeason.roundNames[iRound], power1["offense"], power1["defense"])
               if (fitResults[iSeason].get(game.team2Object.teamId) == None):
                  fitResults[iSeason][game.team2Object.teamId] = [] # array of games for each team
               team2Results = {}
               power2 = game.team2Object.GetPower(thisGameSeason.seasonYear)
               team2Results["roundName"] = thisGameSeason.roundNames[iRound]
               team2Results["roundIndex"] = iRound
               team2Results["defenseFit"] = power2["defense"]
               team2Results["offenseFit"] = power2["offense"]
               fitResults[iSeason][game.team2Object.teamId].append(team2Results)
               game.team2Object.SetPowerActual(thisGameSeason.seasonYear, power2["offense"], power2["defense"])
               game.team2Object.SetFitPower(thisGameSeason.seasonYear, thisGameSeason.roundNames[iRound], power2["offense"], power2["defense"])
      # Populate next season's initial power with the current season's final fit power.
      if (useTeamListFile != None and useTeamSeasonsFile != None and iSeason > 0 and iSeason < len(gameSeasons)):
         activeTeams = team.Team.GetListOfTeamIds(thisGameSeason.seasonYear)
         for teamId in activeTeams:
            teamObject = team.Team.teamObjectById[teamId]
            power = teamObject.GetPowerActual(thisGameSeason.seasonYear)
            teamObject.SetPower(thisGameSeason.seasonYear+1, power["offense"], power["defense"])

   # tmp - print out some results
   #print("::: team-25 fit powers: "+str(fitResults[iSeason][25]))

if (optimizeWeekWeights):
   # Find the best weights to use for each week.  adjustedDef = weight[iGameCount]*fitDef + (1-weight[iGameCount]*prevYearDef) 
   #                                              adjustedOff = weight[iGameCount]*fitOff + (1-weight[iGameCount]*prevYearOff) 
   #
   # cost[iGameCount] = sum over games with iGameCount (weight[iGameCount]*fitDef+(1.-weight[iGameCount])*prevDef - actualDef)**2
   #     (weight[iGameCount]*fitOff+(1.-weight[iGameCount])*prevOff - actualOff)**2
   #
   # Don't worry for now about scaling defense differences with offense. Treat them the same for simplicity.
   #
   # d cost[iGameCount]/d weight[iGameCount] = sum over games with iGameCount 
   #        2*(weight[iGameCount]*fitDef+(1.-weight[iGameCount])*prevYearDef - actualDef)*(fitDef-prevYearDef)
   #      + 2*(weight[iGameCount]*fitOff+(1.-weight[iGameCount])*prevYearOff - actualOff)*(fitOff-prevYearOff)

   adamParams = {}
   adamParams["gameWeeks"] = []
   adamParams["beta1"] = 0.9
   adamParams["beta2"] = 0.999
   adamParams["epsilon"] = 1e-8
   adamParams["beta1t"] = 1.0
   adamParams["beta2t"] = 1.0
   for iGameCount in range(gameSimulator.maxWeekWeightCount):
      gameWeek = {}
      gameWeek["m"] = 0
      gameWeek["v"] = 0
      adamParams["gameWeeks"].append(gameWeek)

   for iWeekWeightIter in range(numberOfIterationsForWeekWeights):

      if (adamUpdateEnable):
         # Add one power of t since doing one time step
         adamParams["beta1t"] *= adamParams["beta1"]
         adamParams["beta2t"] *= adamParams["beta2"]

      thisCost = []
      thisCost = [0 for i in range(gameSimulator.maxWeekWeightCount)]
      thisCostCount = []
      thisCostCount = [0 for i in range(gameSimulator.maxWeekWeightCount)]
      deltaWeights = []
      deltaWeights = [0 for i in range(gameSimulator.maxWeekWeightCount)]

      for iSeason in range(1,numberOfSeasons):
         thisGameSeason = gameSeasons[iSeason]
         for teamId in fitResults[iSeason]:
            finalFitWeek = len(fitResults[iSeason][teamId]) - 1
            for iGameCount in range(finalFitWeek-1):
               weekWeight = gameSimulator.weekWeights[iGameCount]
               fitDef = fitResults[iSeason][teamId][iGameCount]["defenseFit"]
               fitOff = fitResults[iSeason][teamId][iGameCount]["offenseFit"]
               actualDef = fitResults[iSeason][teamId][finalFitWeek]["defenseFit"]
               actualOff = fitResults[iSeason][teamId][finalFitWeek]["offenseFit"]
               prevYearPower = team.Team.teamObjectById[teamId].GetPowerActual(thisGameSeason.seasonYear-1)
               if (prevYearPower["offense"] == None):
                  prevYearPower = GetDefaultPower(team.Team.teamObjectById[teamId].GetDivisionName(year))
               diffDef = weekWeight*fitDef + (1.-weekWeight)*prevYearPower["defense"] - actualDef
               diffOff = weekWeight*fitOff + (1.-weekWeight)*prevYearPower["offense"] - actualOff
               thisCost[iGameCount] += diffDef**2 + diffOff**2
               thisCostCount[iGameCount] += 1
               deltaWeights[iGameCount] += 2*diffDef*(fitDef - prevYearPower["defense"])
               deltaWeights[iGameCount] += 2*diffOff*(fitOff - prevYearPower["offense"])

      for iGameCount in range(gameSimulator.maxWeekWeightCount):
         if (thisCostCount[iGameCount] > 0):
            thisCost[iGameCount] /= thisCostCount[iGameCount]
            deltaWeights[iGameCount] /= thisCostCount[iGameCount]
            if (adamUpdateEnable):
               adamParams["gameWeeks"][iGameCount]["m"] = (adamParams["beta1"]*adamParams["gameWeeks"][iGameCount]["m"]
                  + (1.0 - adamParams["beta1"])*deltaWeights[iGameCount])
               adamParams["gameWeeks"][iGameCount]["v"] = (adamParams["beta2"]*adamParams["gameWeeks"][iGameCount]["v"]
                  + (1.0 - adamParams["beta2"])*deltaWeights[iGameCount]*deltaWeights[iGameCount])
               deltaWeights[iGameCount] = ((adamParams["gameWeeks"][iGameCount]["m"] / (1.-adamParams["beta1t"]))
                  / (math.sqrt(adamParams["gameWeeks"][iGameCount]["v"] / (1.-adamParams["beta2t"])) 
                  + adamParams["epsilon"]))
            gameSimulator.weekWeights[iGameCount] -= learningRateInitial * deltaWeights[iGameCount]
            if (gameSimulator.weekWeights[iGameCount] > 1.):
               gameSimulator.weekWeights[iGameCount] = 1.
            if (gameSimulator.weekWeights[iGameCount] < 0.):
               gameSimulator.weekWeights[iGameCount] = 0.

      print(" iWeekWeightIter "+str(iWeekWeightIter)+" cost "+str(thisCost))
      print(" iWeekWeightIter "+str(iWeekWeightIter)+" weight "+str(gameSimulator.weekWeights))

numProbBins = 5
if (trackLinearPredictions):
   predictions = []
   #for thisGameSeason in gameSeasons:
   iPredOffset = 0
   if (minimalFirstRound):
      iPredOffset = -1
   for iSeason in range(numberOfSeasons):
      if (not minimalFirstRound or iSeason > 0):
         thisGameSeason = gameSeasons[iSeason]
         predictions.append({})
         for iProbBin in range(numProbBins+1):
            predictions[iSeason+iPredOffset][iProbBin] = {}
            predictions[iSeason+iPredOffset][iProbBin]["expected"] = 0
            predictions[iSeason+iPredOffset][iProbBin]["actual"] = 0
            predictions[iSeason+iPredOffset][iProbBin]["total"] = 0
         gameCountByTeamId = {}
         for iRound in range(thisGameSeason.roundCount):
            if (thisGameSeason.seasonInfo.get(thisGameSeason.roundNames[iRound]) != None):
               for game in thisGameSeason.seasonInfo[thisGameSeason.roundNames[iRound]]:
                  # TODO: add this info to team info
                  if (gameCountByTeamId.get(game.team1Object.teamId) == None):
                     gameCountByTeamId[game.team1Object.teamId] = {}
                     gameCountByTeamId[game.team1Object.teamId]["gameCount"] = 0
                     gameCountByTeamId[game.team1Object.teamId]["countByRoundName"] = {}
                  gameCountByTeamId[game.team1Object.teamId]["gameCount"] += 1
                  gameCountByTeamId[game.team1Object.teamId]["countByRoundName"][thisGameSeason.roundNames[iRound]] = gameCountByTeamId[game.team1Object.teamId]["gameCount"]
                  prevSeasonPower1 = game.team1Object.GetPowerActual(thisGameSeason.seasonYear-1)
                  if (prevSeasonPower1["offense"] == None):
                     prevSeasonPower1 = GetDefaultPower(game.team1Object.GetDivisionName(year))
                  iRoundPrev = iRound - 1
                  while (iRoundPrev >= 0 and game.team1Object.GetFitPower(thisGameSeason.seasonYear, thisGameSeason.roundNames[iRoundPrev]) == None):
                     #print("::: iter iRoundPrev "+str(iRoundPrev)+" fitPower "+str(game.team1Object.GetFitPower(thisGameSeason.seasonYear, thisGameSeason.roundNames[iRoundPrev]) == None)))
                     iRoundPrev -= 1
                  if (gameCountByTeamId[game.team1Object.teamId]["gameCount"]-1 < 0 or iRoundPrev < 0):
                     # First game, so use the previous season's power.
                     useDef1 = prevSeasonPower1["defense"]
                     useOff1 = prevSeasonPower1["offense"]
                  else:
                     # Use the fit power from the previous round.
                     fitPower1 = game.team1Object.GetFitPower(thisGameSeason.seasonYear, thisGameSeason.roundNames[iRoundPrev])
                     weight1 = gameSimulator.weekWeights[gameCountByTeamId[game.team1Object.teamId]["gameCount"]-1]
                     useDef1 = weight1*fitPower1["defense"] + (1.-weight1)*prevSeasonPower1["defense"]
                     useOff1 = weight1*fitPower1["offense"] + (1.-weight1)*prevSeasonPower1["offense"]
                  if (gameCountByTeamId.get(game.team2Object.teamId) == None):
                     gameCountByTeamId[game.team2Object.teamId] = {}
                     gameCountByTeamId[game.team2Object.teamId]["gameCount"] = 0
                     gameCountByTeamId[game.team2Object.teamId]["countByRoundName"] = {}
                  gameCountByTeamId[game.team2Object.teamId]["gameCount"] += 1
                  gameCountByTeamId[game.team2Object.teamId]["countByRoundName"][thisGameSeason.roundNames[iRound]] = gameCountByTeamId[game.team2Object.teamId]["gameCount"]
                  prevSeasonPower2 = game.team2Object.GetPowerActual(thisGameSeason.seasonYear-1)
                  if (prevSeasonPower2["offense"] == None):
                     prevSeasonPower2 = GetDefaultPower(game.team2Object.GetDivisionName(year))
                  iRoundPrev = iRound - 1
                  while (iRoundPrev >= 0 and game.team2Object.GetFitPower(thisGameSeason.seasonYear, thisGameSeason.roundNames[iRoundPrev]) == None):
                     iRoundPrev -= 1
                  if (gameCountByTeamId[game.team2Object.teamId]["gameCount"]-1 < 0 or iRoundPrev < 0):
                     # First game, so use the previous season's power.
                     useDef2 = prevSeasonPower2["defense"]
                     useOff2 = prevSeasonPower2["offense"]
                  else:
                     # Use the fit power from the previous round.
                     fitPower2 = game.team2Object.GetFitPower(thisGameSeason.seasonYear, thisGameSeason.roundNames[iRoundPrev])
                     weight2 = gameSimulator.weekWeights[gameCountByTeamId[game.team2Object.teamId]["gameCount"]-1]
                     useDef2 = weight2*fitPower2["defense"] + (1.-weight2)*prevSeasonPower2["defense"]
                     useOff2 = weight2*fitPower2["offense"] + (1.-weight2)*prevSeasonPower2["offense"]
                  scoreAndProb = gameSimulator.GetScoreAndProbability(useOff1, useDef1, useOff2, useDef2, game.homeField)
                  probability = scoreAndProb["probability"]
                  if (probability < 0.5):
                     probability = 1. - probability
                  iProbBin = int(2*numProbBins*(probability-0.5))
                  predictions[iSeason+iPredOffset][iProbBin]["expected"] += probability
                  predictions[iSeason+iPredOffset][iProbBin]["total"] += 1
                  # last bin, numProbBins, is for total across all games
                  predictions[iSeason+iPredOffset][numProbBins]["expected"] += probability
                  predictions[iSeason+iPredOffset][numProbBins]["total"] += 1
                  predictions[iSeason+iPredOffset][numProbBins]["year"] = thisGameSeason.seasonYear
                  #print("::: spreadPred "+str(scoreAndProb["score1"]-scoreAndProb["score2"])+" spreadActual "+str(game.score1-game.score2))
                  if ((scoreAndProb["score1"]-scoreAndProb["score2"]) * (game.score1-game.score2) > 0.):
                     # result matches prediction
                     predictions[iSeason+iPredOffset][iProbBin]["actual"] += 1
                     predictions[iSeason+iPredOffset][numProbBins]["actual"] += 1
   print("::: predictions "+str(predictions))

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

