# Minigame_Hit_and_Run
Design of an agent that exploits cliff vaulting mechanics of Colossus in Starcraft II.  



## Map Information:

One Colossus stands on top a cliff, while 27 Zerglings converge to attack on the Colossus climbing from ramps that are located from the sides of the cliff. The aim is to develop an agent which exploits cliff vaulting mechanics of Colossus in order to develop an advantage over Zerglings. The test time is set to 5 minutes and the game ends either when the Colossus ends or the dedicated time period has finished. The game resets itself while retaining reward points and the health of the Colossus, when all Zerglings on the map are killed.

### Initial State:
- 1 agent controlled Colossus
- 27 computer controlled Zerglings

### Rewards: (tentative)
+ +2 for killing each Zergling
+ +x points for every x health that Colosssus has (excluding shields)
+ -20 for dying

### End Conditions:
- Time limit reached
- Colossus dies

### Time Limit:
- 300 seconds

### Notes:
- Fog of War disabled
- No upgrades available
- The policy might learn micro strategy of kiting the enemy units by using cliff vault mechanic. Timing is crucial for agent to optimize Colossus health while inflicting most damage.
