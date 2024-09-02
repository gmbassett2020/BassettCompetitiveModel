
#
# Game class
#
              
# uses Team class

class Game:
   """Class to hold info on a game"""

   nextGameId = 0
   gameObjectById = {}
    
   def __init__(self, team1Object, team2Object, roundName, homeField, score1=None, score2=None):
      self.team1Object = team1Object
      self.team2Object = team2Object
      self.roundName = roundName
      self.homeField = homeField
      self.score1 = score1
      self.score2 = score2
      self.gameId = Game.nextGameId
      Game.nextGameId += 1
      Game.gameObjectById[self.gameId] = self    
        
   def SetResults(self, score1, score2):
      self.score1 = score1
      self.score2 = score2   
    
   def PrintGame(self, headerString=""):
      teamString = ""
      resultString = ""
      #print("::: homeField="+str(self.homeField)+" team1="+self.team1Object.teamName+" team2="+self.team2Object.teamName)
      if self.homeField == 1:
         teamString += '{0} at {1}'.format(self.team1Object.teamName, self.team2Object.teamName)
         if (self.score1 != None):
            resultString = ' {0}-{1}'.format(str(self.score1), str(self.score2))
      elif self.homeField == 0:
         teamString += '{0} at {1}'.format(self.team2Object.teamName, self.team1Object.teamName)
         if (self.score1 != None):
            resultString = ' {0}-{1}'.format(str(self.score2), str(self.score1))
      else:
         teamString += '{0} vs {1}'.format(self.team1Object.teamName, self.team2Object.teamName)
         if (self.score1 != None):
            resultString = ' {0}-{1}'.format(str(self.score1), str(self.score2))
      print(headerString+teamString+resultString)

