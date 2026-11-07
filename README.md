# pfa-week10
Fiori Sims Week 10 Homework for Programming for Animators FA26

## Reference World Builder

An Unreal Engine 5.8 tool that turns a concept painting or layered Photoshop file into a walkable 3D
world. It finds the objects in the picture, places each one where it was painted, builds a new 3D model
from each outline, and paints the models, ground and water with the painting's own colours and
brushstrokes.

Full feature notes are in [`ReferenceWorldBuilder/README.md`](ReferenceWorldBuilder/README.md).

---

## What you need

| | Version | Needed for |
| --- | --- | --- |
| **Windows 10/11** | | Unreal and the helpers (macOS/Linux untested) |
| **Unreal Engine** | **5.8** (Epic Games Launcher) | everything |
| **Python** | **3.12**, on your `PATH` | analysing pictures, building models |
| **Blender** | 4.x or 5.x (tested on 5.2) | recommended: cleaner models, brushstrokes on every side, *Find model* |
| **Adobe Photoshop** | 2026 | only for `.psd` / `.psb` files |
| **Internet** | | first run only (vision models), and *Find model* |
| Memory | 16 GB RAM | builds use about 4 GB on top of Unreal |

No C++ compiler or Visual Studio is needed.

## Setting up on a new computer

1. **Download this repo.** Click **Code > Download ZIP** and unzip it, or clone it:
   ```
   git clone https://github.com/fiorisims/pfa-week10.git
   ```
   Keep the folder layout as it is: the Unreal project (`ReferenceWorldBuilder/`) must sit next to
   the Python scripts, because the tool looks for them in the folder above the project.

2. **Install Python 3.12** from [python.org](https://www.python.org/downloads/). In the installer,
   tick **"Add python.exe to PATH"**. Check it in a new terminal:
   ```
   python --version
   ```

3. **Install the Python packages.** In a terminal, inside the downloaded folder:
   ```
   python -m pip install -r requirements.txt
   ```

4. **Install Blender** (optional but recommended) from [blender.org](https://www.blender.org/download/).
   The tool finds it in `C:\Program Files\Blender Foundation\` automatically. If you installed it
   somewhere else, set an environment variable `BLENDER_EXE` to the full path of `blender.exe`.

5. **Install Unreal Engine 5.8** from the Epic Games Launcher.

6. **Check the setup (optional).** This runs the tool's own checks without Unreal, in about a minute:
   ```
   python test_reference_world.py
   ```
   It should end with `all checks passed`.

## Running the tool

1. **Open the project:** double-click `ReferenceWorldBuilder/ReferenceWorldBuilder.uproject`.
   The first time, Unreal compiles shaders for a few minutes. The **Reference World Builder** panel
   opens by itself (or use **Tools > Open Reference World Builder**). It may open behind other windows.

2. **Pick your picture:** click **Browse** and choose a PNG, JPG, PSD or PSB.
   The world is generated straight away. A progress window shows each step.
   - The **first analysis downloads the vision models (about 350 MB)**, once.
   - A first build takes about 3-5 minutes; generating again takes under a minute.
   - Unreal can't be used while the progress window is up. That's normal.

3. **Walk around:** press **Play in editor**. You start where the painter's eye was, looking into the
   picture. Walk into the water: you wade through it.

4. **Save a result:** press **Ctrl+S** to save the level. **Clear** removes everything the tool made.

## Making the models look right

Everything here is in the panel's **MODELS** list, one row per model:

- **- / + / +5** - how many of that model are in the world. Click its **name** to select it in the level.
- **Shape** - cycles how it's built: Puffy, Round, Flat front, Leafy mass, Building, Lying down.
- **Redraw outline** - opens the drawing window with its outline, so you can fix it.
- **What is it? + Find model** - type what the object really is (e.g. *birdcage*, *wooden canoe*),
  press **Find model**, then click the preview picture you like. It searches Poly Haven, Objaverse
  and your `ModelLibrary` folder, and fits the model you pick to your painting.
  **Use painted** goes back.
- **Draw a new shape...** - draw on your painting to add a new object exactly where you draw it.

Other settings in the **BUILD** section:

- **Model style** - 0 = generic low poly, 1 = as painted (press **Generate world** after moving it).
- **Brushstrokes** - how strong your paint marks are on the sides you never painted.
- **Grow grass on grassy ground**, **World size**, **Terrain height**, **Prop density**, **Seed**.

## Tips for the best results

- **Layered PSDs work best.** Name layers after what they are (`Tree`, `Tower`, `Lake`, `Sky`); add
  `FG` / `MG` / `BG` to say how far away something is. Layers called "Layer 3" are recognised from
  their pixels.
- Drop your own models (`.glb`, `.gltf`, `.fbx`, `.obj`) into `ModelLibrary/` and they can be found by
  file name.
- Close other big programs (browsers, chat apps) before a build if memory is tight.

## If something goes wrong

| Problem | Fix |
| --- | --- |
| "Python with pygame was not found" | Install Python with **Add to PATH**, then run step 3 again. |
| Nothing happens after Browse | Look for the progress window behind other windows; check **Window > Output Log** for lines starting `Reference World Builder:`. |
| Models look grey or white | Close and reopen the project once. |
| "Blender not found" in the panel | Install Blender, or set `BLENDER_EXE`. The tool still works without it, with plainer models. |
| A PSD won't open | Photoshop must be installed; the first PSD can take a minute while Photoshop starts. |
| *Find model* finds nothing | It needs the internet. Try other words: some things (castles, gazebos) aren't in the libraries. |

## Credits and licences

- Nature models: [Kenney Nature Kit](https://kenney.nl/assets/nature-kit) (CC0).
- Vision models, downloaded on first use: OWLv2, SlimSAM, CLIP (via Hugging Face), Depth Anything V2.
- *Find model* uses [Poly Haven](https://polyhaven.com) (CC0) and [Objaverse](https://objaverse.allenai.org/)
  (only CC0 and CC-BY models). Models you use are listed with their authors in
  `ModelLibrary/_online/CREDITS.txt`. **CC-BY models must be credited if you share or publish work that uses them.**
