# Research Project 1 - Proposal

**What is the project, in one sentence?**
Reference World Builder is an Unreal Engine tool that turns a concept painting
or Photoshop file into a small playable environment, using the image's palette
for lighting, its brightness for terrain height and its colour regions to place
trees, rocks and flowers.

**Why this one?**
It connects things I already build - my Photoshop palette and color-script
tools - to Unreal, which is new to me as a scripting target. As a concept artist,
I want to walk around inside a painting early, before committing to a full
environment build.

**What will you show at the end?**
A live demo: import one of my own paintings, generate the world, then play
through it with props that react as the character walks near them, plus a short
recorded walkthrough comparing the painting with the generated scene.

**Three weeks**
- **Nov 10:** Image/PSD analysis working and feeding Unreal; terrain generated from brightness.
- **Nov 17:** Palette-driven lighting, water and colour-based prop placement; Play-mode proximity reactions.
- **Nov 23:** Polish the panel, test with several of my own paintings, record the demo.

**Smallest version that still counts**
Pick an image, press one button, and get a terrain whose shape and colours come
from that image that I can walk on in Play mode.

**Riskiest part**
Unreal's Python can't open file dialogs or load my image libraries, so it has
to talk to a separate Python process. Making that hand-off reliable, and making
the result look intentional rather than random, are the hard parts.
