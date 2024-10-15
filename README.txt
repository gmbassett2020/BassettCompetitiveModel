The Bassett Competitive Model is a set of tools to make predictions and rankings for teams playing a sport which has a score for each team.  It has been used for NCAA football and NFL for the 2023 season of the Bassett Football Model.  The competitive model is based on the Bassett Football Model, putting down in simplified form the characteristics of that earlier model.

Bassett Football Model links:
   + General description and current forecasts & rankings: http://BassettFootball.net
   + Bassett Competitive Model results for 2023 season:
      - http://BassettFootball.net/col_23.html
      - http://BassettFootball.net/nfl_23.html

Python programs:
   + forecastAndRank.py
      - Run forecast and rankings for a given week.
      - For usage, see //https://github.com/gmbassett2020/BCMWorkspace
   + createAveragePowerStats.py
      - Given a set of fit team powers for several seasons, compute the average, RMS and change between seasons for each league and division.
      - Primary input parameters: useTeamListFile, inputPowerFiles, outputPowerFile
      - Statistics are printed to STDOUT.
   + simulatedFootball.py & simulateOneSeason.py
      - Generate synthetic football seasons to test/fit parameters.
      - Initial development: simulatedFootball.ipynb

Help on module gameSeason:

NAME
    gameSeason

DESCRIPTION
    # Develop simulated football season.
    # Connect to a Java Neural Network using Lasagne.
    # http://lasagne.readthedocs.org/
    #    More in-depth examples and reproductions of paper results are maintained in
    #    a separate repository: https://github.com/Lasagne/Recipes
    # 2022-01-17
    # 2022-08-05
    #  Moved from jupyter-notebook to python file.
    #  Removing neural net items since focusing first on purely linear model.

CLASSES
    builtins.object
        GameSeason

    class GameSeason(builtins.object)
     |  GameSeason(seasonYear, gameSimulator=None, name='')
     |
     |  Class for tracking team matchups for one season
     |
     |  Methods defined here:
     |
     |  AddGame(self, team1, team2, roundName, homeField, generateScore=None, score1=None, score2=None)
     |
     |  AdjustPowerAndLinearParametersFromScores(self, learningRate, adamUpdateEnable, parameters=None, averageTeamPower=15.37, powerCostFactor=0.001, verbose=False, constrainAveragePower=True, forceAverageOffense=None, forceAverageDefense=None, roundStopCount=None)
     |
     |  CreateSimulatedSeason(self, seasonType='NCAA2Divsions', correlatePreviousSeasonsPower=True, initializePowerFromPrevious=False, uniqueTeamNames=False)
     |
     |  FitPowerForCurrentRound(self, learningRate=0)
     |
     |  GetAveSeasonChange(divisionName)
     |
     |  GetCurrentPower(self, teamObject, year, roundName)
     |
     |  PrintSeason(self)
     |
     |  PrintTeamSchedules(self)
     |
     |  RandomizeTeamPower(self, maxOffense, minOffense, maxDefense, minDefense)
     |
     |  SetTeamSchedules(self)
     |
     |  __init__(self, seasonYear, gameSimulator=None, name='')
     |      Initialize self.  See help(type(self)) for accurate signature.
     |
     |  ----------------------------------------------------------------------
     |  Static methods defined here:
     |
     |  GetDefaultPower(divisionName)
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |
     |  __dict__
     |      dictionary for instance variables
     |
     |  __weakref__
     |      list of weak references to the object
     |
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |
     |  defAve = [7.417854504803192, 5.613700341866554, 6.717104255447838, 6.7...
     |
     |  defDivRms = [1.2529964382387893, 1.2206797283294453, 0.733423609069076...
     |
     |  defSeasonChange = [1.8424067896960112, 1.955, 1.3057518675418245, 1.30...
     |
     |  divName = ['IA', 'IAA', 'NFC', 'AFC']
     |
     |  nflDefAve = 6.717104255447838
     |
     |  nflDefDelta = 1.3057518675418245
     |
     |  nflDefRms = 0.7334236090690767
     |
     |  nflOffAve = 8.721953466149213
     |
     |  nflOffDelta = 1.0994463357571216
     |
     |  nflOffRms = 0.6696469160380194
     |
     |  offAve = [10.264432962217683, 7.694884091447714, 8.721953466149213, 8....
     |
     |  offDivRms = [1.1057786122920785, 1.341083431558368, 0.6696469160380194...
     |
     |  offSeasonChange = [1.5770794762076497, 1.686645465207797, 1.0994463357...

FILE
    /mnt/linux4tb2024/bcm-wrk/bcm/gameSeason.py


Help on module gameSimulator:

NAME
    gameSimulator

CLASSES
    builtins.object
        GameSimulator

    class GameSimulator(builtins.object)
     |  GameSimulator(homeFieldAdvantage, spdCoefOff, spdCoefDef, totCoef0, totCoefOff, totCoefDef, roundScores, spreadRms, maxConfGameWeights)
     |
     |  Class for generating results/predictions of games
     |
     |  Methods defined here:
     |
     |  AdjustPowerToFitActualScore(self, o1, d1, o2, d2, homeField, actualScore1, actualScore2)
     |
     |  ApplyDeltaWeights(self)
     |
     |  GetRandomizedScore(self, o1, d1, o2, d2, homeField)
     |
     |  GetScoreAndProbability(self, o1, d1, o2, d2, homeField, dO1=None, dD1=None, dO2=None, dD2=None)
     |
     |  GetScores(self, o1, d1, o2, d2, homeField, verbose=False)
     |
     |  GetTotalNoSpread(self, o1, d1, o2, d2, actualScore1, actualScore2, homeField, verbose=False)
     |
     |  RandomizeLinearParameters(self, randomizeHomeFieldAdvantage=True, randomizeLinearParameters=True)
     |
     |  RandomizeWeights(self)
     |
     |  ResetDeltaWeights(self)
     |
     |  SetWeekWeightsToBestFit(self)
     |
     |  SetWeights(self, value=1)
     |
     |  SetWeightsToBestFit(self)
     |
     |  __init__(self, homeFieldAdvantage, spdCoefOff, spdCoefDef, totCoef0, totCoefOff, totCoefDef, roundScores, spreadRms, maxConfGameWeights)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |
     |  __dict__
     |      dictionary for instance variables
     |
     |  __weakref__
     |      list of weak references to the object

FILE
    /mnt/linux4tb2024/bcm-wrk/bcm/gameSimulator.py


Help on module game:

NAME
    game

DESCRIPTION
    # Game class
    #

CLASSES
    builtins.object
        Game

    class Game(builtins.object)
     |  Game(team1Object, team2Object, roundName, homeField, score1=None, score2=None)
     |
     |  Class to hold info on a game
     |
     |  Methods defined here:
     |
     |  PrintGame(self, headerString='')
     |
     |  SetResults(self, score1, score2)
     |
     |  __init__(self, team1Object, team2Object, roundName, homeField, score1=None, score2=None)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |
     |  __dict__
     |      dictionary for instance variables
     |
     |  __weakref__
     |      list of weak references to the object
     |
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |
     |  gameObjectById = {}
     |
     |  nextGameId = 0

FILE
    /mnt/linux4tb2024/bcm-wrk/bcm/game.py


Help on module team:

NAME
    team

DESCRIPTION
    # Team class
    #

CLASSES
    builtins.object
        Team

    class Team(builtins.object)
     |  Team(teamName, startYear=None, conferenceName=None, divisionName=None)
     |
     |  Class holding team information
     |
     |  Methods defined here:
     |
     |  GetAdamMoments(self, year, updateBetaTs=True)
     |
     |  GetConferenceName(self, year)
     |
     |  GetDefensePower(self, year)
     |
     |  GetDivisionName(self, year)
     |
     |  GetFitPower(self, year, roundName)
     |
     |  GetHomeField(self, year, roundName)
     |
     |  GetOffensePower(self, year)
     |
     |  GetOpponent(self, year, roundName)
     |
     |  GetPower(self, year)
     |
     |  GetPowerActual(self, year)
     |
     |  SetAdamMoments(self, year, moments)
     |
     |  SetConferenceAndDivision(self, year, conference, division)
     |
     |  SetFitPower(self, year, roundName, offense, defense)
     |
     |  SetOpponent(self, year, roundName, opponentName, homeField)
     |
     |  SetPower(self, year, offensePower, defensePower)
     |
     |  SetPowerActual(self, year, offensePower, defensePower)
     |
     |  __init__(self, teamName, startYear=None, conferenceName=None, divisionName=None)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |
     |  ----------------------------------------------------------------------
     |  Static methods defined here:
     |
     |  GetListOfTeamIds(year)
     |
     |  IsTeamActiveById(teamId, year)
     |
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |
     |  __dict__
     |      dictionary for instance variables
     |
     |  __weakref__
     |      list of weak references to the object
     |
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |
     |  nextTeamId = 0
     |
     |  teamObjectById = {}
     |
     |  teamObjectByName = {}

FILE
    /mnt/linux4tb2024/bcm-wrk/bcm/team.py


