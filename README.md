# Balance the Pencil

A simple Neural Network demonstration.

## The game itself

The aim is to balance the pencil as long as possible. If the pencil falls, or goes of screen, you lose

The physics is two main things: rotation due to gravity, and rotation due to tip movement. There is also some light damping/ air resistance but it doesn't seem to have a major effect and probably doesn't need including if this project were made again.

## How the Network learns

-   The network has 3 inputs: 
	- **angle, tip speed, tip position**
-   The network has 1 output:
	- **movement speed**

Movement speed is between 0 and 1, 0.5 meaning don’t move, 1 meaning move right at full speed.

Each time the pencil falls, the network is randomized and the game restarted. The most successful networks (the ones that balanced the pencil for the longest) are saved for future reference. After a sufficient number of random networks has been tried, the new networks are chosen by adding two of the top 5 most successful networks together, plus some random variation.

This is sort of a crappy version of [NEAT](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies), which mimics how living things evolve. The idea is if we keep choosing the most successful networks to create our new networks, our best networks will converge on the best "solution" to the problem of balancing the pencil.

## Understanding the Network's solution

The best network seems to be one with all positive weights: 

<img src=https://raw.githubusercontent.com/toronyx/balance-the-pencil/main/pictures/good_solution.png width="400">

-   The angle is positively associated with movement, i.e. if the pencil tilts to the right, the network moves to the right.
-   The position of the tip is positively associated with movement. This one is more subtle, if the position of the tip is near the right side of the screen, the network actually moves to the right faster. This is so that the pencil starts tilting to the left, allowing the pencil to be moved back to the centre. Think about it, if the tip of the pencil slowed down as it moved towards the edge, the pencil would fall forwards.
-   The speed is very similar, if the pencil is travelling towards the edge too fast, the network tries to move the tip ahead of the pencil, so the pencil tilts back. Try balancing a pencil and you may understand better.

## Videos

Below are some videos of the game/network in action!

### Gameplay, i.e. playing the game as a human.

https://user-images.githubusercontent.com/83616452/194716412-c07e672c-fdb6-4a9d-aeff-35ce604b94ef.mp4

### Randomly generated networks attempting to balance the pencil

You can see the nodes of the network, which change colour depending on their value. The input nodes are the "observations" about the game world, and the output nodes are the model's "decisions". The video has been sped up a bit so we can see more attempts.

https://user-images.githubusercontent.com/83616452/194716439-8bc1a183-cb57-4152-bf82-7028318ea8d1.mp4

## A successful network

After randomly generating over 100 networks, the model has started to converge on a good solution

https://user-images.githubusercontent.com/83616452/194716707-742fdc75-3f3c-4b78-a3bd-6c8b8e057c7a.mp4

Here I have sped it up so we can see how it does over a longer period of time:

https://user-images.githubusercontent.com/83616452/194716442-1d4d1f73-1b32-4fc8-80de-59e8f5c095ec.mp4
