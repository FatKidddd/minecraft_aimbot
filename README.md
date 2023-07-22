Aimbot tackling a Minecraft server's target practice challenge 

## End result:
The gameplay in the demo vid is half the speed of the actual thing because OBS lags my laptop.

![](resources/final.mp4)

## Context:
Multiple target bottle(s) will pop up with each wave and the player has to only shoot down all target bottles without missing within a short amount of time

Objective is to shoot down bottles both accurately and quickly

## Solution:
#### Image processing to filter out bottles
	- Masked screenshot using pixel color of bottle outline 
	- Removed small white noise using (2,2) kernel erosion - balance between removing noise and small pixelated outline of targets
	- Used connected components to remove noise while maintaining outlines
	- Closed image to combine broken pieces of a target into one target
	- Used contours to get positions of bottles
#### Movement algorithm
	- Determined the core formula to translate screen distances to the required mouse input movement
		- mouse_movement = arctan(distance_on_screen/k) * 180/PI * 0.15, where 1 pixel of movement is equivalent to 0.15 deg rotation in Minecraft
		- Determined k through manual calibration and trial and error
			- Obtained theoretical value of k = 490
			- Experimental value of k = 545 (hacky fix for movement overshooting issue)
		- Rounded down all calculations to the nearest integer (due to floating movement issue)
	- Movement towards closest target
		- Sort all bottles based on their distance towards the crosshair (centre of screen) and always move towards the closest bottle to ensure convergence on bottle
		- Shoot when crosshair is within bottle dimensions for 3 frames (hacky fix for movement overshooting issue)
	- Time-based memory solution to determine targets that have not been shot at before to
		- move towards the next target
		- prevent double shooting (targets take a while to be destroyed after being shot at, worsened by server lag) and
		- overcome (accumulation of movement inaccuracies issue)

		- Track all positions that were shot at up to 2 seconds ago through storing cumulative movements
		- If tracked position and actual position of target (as observed in screenshot) is within threshold --> has been shot at before and thus not included in list of candidate bottles
#### Improving performance (processing time, bot movement, bot accuracy)
	- Reduced screen capture size by having bot move back to the centre of the zone 
		- Bot uses green bush at the top left corner of the pit as a reference for calibration (due to accumulation of movement inaccuracies)
	- Used game update of mouse.position to ensure that Minecraft has fully processed the mouse movement instead of relying on time.sleep delays which are much slower
	- Dark sky at night causes bot to stare it --> excluded sky to be in vision by reducing screen size as bot looks upwards 
	- Experimenting to determine ideal FOV --> settled with FOV of 95 which has a good balance of little image noise and easy recognition of targets 

## Notable problems
#### Image processing issues - finding a balance between reducing noise and allowing easy detection of objects (resolved)
	- Experimented with different processing techniques at different timestamps of developing the bot since image processing, mouse movement, and FOV all the other problems that affected it are largely linked
#### Movement obstacles
	- Failure to converge on target 
		- Theoretical k causes realised movement to overshoot
			- I really don't know why because theoretical k works accurately to track past shots
		- Double movement - occasionally, Minecraft will somehow not yet finish processing mouse movement (movement is not reflected on screen) but the game has updated the mouse position based to the centre of the screen, resulting in similar displacements occuring twice, leading to overshooting
		- Crosshair gets stuck jumping back and forth between two targets
			- Based on the debugged values where calculated values somehow seems to cause the crosshair to move away from target, I suspect the issue is similar to the issue of double movement
	- Floating point movement - <1 pixel width mouse movement is simply considered as 0 movement in Minecraft (resolved by rounding down calculations to integer)
	- Accumulation of movement inaccuracies - cannot have bot retrace all the steps it took to return back to original position, it will either overshoot or undershoot (suspect due to floating point movement issue)
	- Leftover mouse velocity due to mouse acceleration
		- Resolved by disabling "Enhance pointer precision" setting to remove mouse acceleration from causing 'leftover' movements and preventing overshooting
#### Server side lag
	- Changed Minecraft client to improve performance
	- Track past shots for 2 seconds to give server ample time to update shot at target
#### Client side lag, python just freezes up and movement becomes extremely slow (occurs randomly) <-- probably due to low laptop specs
	- Failed to identify potential errors through profiling both time and memory
	- Read up more about multithreading and multiprocessing but not really applicable in this context, bot needs to wait for finished movement before grabbing screen
	- Switched to Fabric in hopes of improving Minecraft performance
	- Attempted to convert to Cython but didn't have enough time with few hours of event left

## Final results
Bot averages 150+ waves per attempt but dies to client side lag most of the time or server side lag (very occasionally there is a huge lag spike).
My average performance is 30+ waves.
