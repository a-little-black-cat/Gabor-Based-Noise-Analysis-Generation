"""
1. Implement gabor filter
    1.a: set filter parameters 
    1.b: set type of texture (grass, gravel, brick)
    1.c: set size of generated image
    1.d: set count of generated images
2. Compute features of the image using gabor filter
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.filters import gabor_kernel
import tkinter as Tk
import csv
import queue
import threading
from tkinter import messagebox, ttk, filedialog
from PIL import Image, ImageTk
import gabor_extractor as extractor
from scipy.fft import fft2, fftshift
from scipy.optimize import curve_fit


# prompt: Text shown to the user.
# default: Value used when the user submits an empty response.
# converter: Callable that converts the entered text to the required type.
def prompt_value(prompt, default, converter):
    value = input(f"{prompt} [{default}]: ").strip()
    return default if not value else converter(value)
    # This is to minimize the number of input prompts for the user. If the user presses enter without typing anything, the default value will be used. Otherwise, the input will be converted to the specified type (int or float) using the provided converter function.

# No parameters; creates and runs the texture generator UI.
def userInputMode():
    root = Tk.Tk()
    root.title("Gabor Texture Generator")
    root.geometry("620x760")

    outer_frame = ttk.Frame(root, padding=12)
    outer_frame.pack(fill="both", expand=True)

    canvas = Tk.Canvas(outer_frame, highlightthickness=0)
    scrollbar = ttk.Scrollbar(outer_frame, orient="vertical", command=canvas.yview)
    form_frame = ttk.Frame(canvas)
    canvas_window = canvas.create_window((0, 0), window=form_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # _event: Optional Tkinter configure event that triggers the update.
    def update_scroll_region(_event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))

    # event: Tkinter canvas resize event used to match the form width.
    def resize_form(event):
        canvas.itemconfigure(canvas_window, width=event.width)

    form_frame.bind("<Configure>", update_scroll_region)
    canvas.bind("<Configure>", resize_form)

    ttk.Label(form_frame, text="Gabor Texture Generator", font=("TkDefaultFont", 16, "bold")).pack(anchor="w", pady=(0, 12))

    general_defaults = {
        "height": 512,
        "width": 512,
        "variation_size": 512,
        "image_count": 1,
        "texture_rotation": 0.0,
        "seed": 42,
    }
    brick_defaults = {
        "brick_width": 64,
        "brick_height": 32,
        "mortar_width": 2,
        "mortar_height": 2,
        "frequency": 0.08,
        "theta": 0.0,
        "sigma_x": 5.0,
        "sigma_y": 2.0,
        "vertical_frequency": 0.08,
        "vertical_theta": 90.0,
        "vertical_sigma_x": 5.0,
        "vertical_sigma_y": 2.0,
        "horizontal_weight": 0.7,
        "vertical_weight": 0.3,
        "mortar_frequency": 0.12,
        "mortar_theta": 0.0,
        "mortar_sigma_x": 2.0,
        "mortar_sigma_y": 1.0,
        "brick_base": 0.55,
        "brick_contrast": 0.2,
        "mortar_base": 0.25,
        "mortar_contrast": 0.05,
    }
    
    grass_defaults = {
        "frequency": 0.055,
        "theta": 0.0,
        "sigma_x": 7.0,
        "sigma_y": 1.2,
    }

    gravel_defaults = {
        "frequency": 0.08,
        "theta": 0.0,
        "sigma_x": 5.0,
        "sigma_y": 2.0,
    }

    rock_defaults = { #LIKELY to be converted into woronoi tessellation or something similar -- might be combined with proc. rock generator in future
        "frequency": 0.08,  
        "theta": 0.0,
        "sigma_x": 5.0,
        "sigma_y": 2.0,
    }


    variables = {}
    uploaded_parameters = None
    uploaded_image = None
    variation_window = None
    processing_window = None

    # title: Text displayed in the section's label frame.
    def add_section(title):
        section = ttk.LabelFrame(form_frame, text=title, padding=8)
        section.pack(fill="x", pady=(0, 10))
        return section

    # section: Parent Tkinter frame for the input field.
    # name: Internal key used to store the field variable.
    # label: Text displayed beside the input field.
    # default: Initial value displayed in the input field.
    def add_field(section, name, label, default):
        row = ttk.Frame(section)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label).pack(side="left", anchor="w")
        variable = Tk.StringVar(value=str(default))
        variables.setdefault(name, []).append(variable)
        ttk.Entry(row, textvariable=variable, width=14).pack(side="right")

    # No parameters; displays the modal processing indicator.
    def show_processing_window():
        nonlocal processing_window
        if processing_window is not None and processing_window.winfo_exists():
            processing_window.destroy()

        processing_window = Tk.Toplevel(root)
        processing_window.title("Processing")
        processing_window.geometry("280x120")
        processing_window.resizable(False, False)
        processing_window.transient(root)
        processing_window.grab_set()
        ttk.Label(processing_window, text="Processing...", font=("TkDefaultFont", 14, "bold")).pack(pady=(22, 8))
        progress = ttk.Progressbar(processing_window, mode="indeterminate", length=210)
        progress.pack()
        progress.start(10)
        processing_window.protocol("WM_DELETE_WINDOW", lambda: None)
        root.update_idletasks()

    # No parameters; closes the processing indicator if it exists.
    def hide_processing_window():
        nonlocal processing_window
        if processing_window is not None and processing_window.winfo_exists():
            processing_window.grab_release()
            processing_window.destroy()
        processing_window = None

    # No parameters; opens a file picker and starts texture extraction.
    def upload_texture():
        nonlocal uploaded_parameters, uploaded_image
        file_path = filedialog.askopenfilename(
            title="Select Texture Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff")],
        )
        if file_path:
            print(f"Selected texture image: {file_path}")
            show_processing_window()
            result_queue = queue.Queue()

            # No parameters; extracts parameters and image data in the worker thread.
            def process_upload():
                try:
                    extracted = extractor.extract_gabor_parameters(file_path)
                    image = np.asarray(
                        Image.open(file_path).convert("RGB"), dtype=np.float32
                    ) / 255.0
                    result_queue.put((extracted, image, None))
                except Exception as error:
                    result_queue.put((None, None, error))

            # No parameters; checks for worker results and updates Tkinter widgets.
            def finish_upload():
                nonlocal uploaded_parameters, uploaded_image
                try:
                    extracted, image, error = result_queue.get_nowait()
                except queue.Empty:
                    root.after(100, finish_upload)
                    return

                hide_processing_window()
                if error is not None:
                    messagebox.showerror("Texture extraction failed", str(error), parent=root)
                    status.set("Texture extraction failed")
                    return

                uploaded_parameters = extracted
                uploaded_image = image
                for name in ("frequency", "sigma_x", "sigma_y"):
                    for variable in variables[name]:
                        variable.set(f"{extracted[name]:.6g}")
                for variable in variables["theta"]:
                    variable.set(f"{np.rad2deg(extracted['theta']):.3f}")
                status.set(
                    f"Extracted {len(extracted['components'])} noise components "
                    f"from {extracted['image_size']['width']}x{extracted['image_size']['height']} image"
                )
                print(extracted)

            threading.Thread(target=process_upload, daemon=True).start()
            root.after(100, finish_upload)

    # textures: Sequence of generated RGB texture arrays with values from 0.0 to 1.0.
    def show_variations(textures):
        nonlocal variation_window
        if variation_window is not None and variation_window.winfo_exists():
            variation_window.destroy()

        variation_window = Tk.Toplevel(root)
        variation_window.title("Uploaded Texture Variations")
        variation_window.geometry("980x760")
        variation_window.columnconfigure(0, weight=1)
        variation_window.rowconfigure(0, weight=1)
        content_frame = ttk.Frame(variation_window)
        content_frame.grid(row=0, column=0, sticky="nsew")
        content_frame.columnconfigure(0, weight=1)
        content_frame.rowconfigure(1, weight=1)
        header_frame = ttk.Frame(content_frame)
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew")

        #get extracted parameters from uploaded image and display them in the variation window
        if uploaded_parameters:
            for i, param in enumerate(uploaded_parameters["components"]):
                ttk.Label(header_frame, text=f"Component {i + 1}: Frequency={param['frequency']:.6g}, Theta={np.rad2deg(param['theta']):.3f}, Sigma_X={param['sigma_x']:.6g}, Sigma_Y={param['sigma_y']:.6g}").grid(row=i, column=0, sticky="w", padx=12)

            palette_frame = ttk.LabelFrame(header_frame, text="Extracted Color Palette", padding=8)
            palette_frame.grid(row=len(uploaded_parameters["components"]), column=0, sticky="ew", padx=12, pady=(8, 12))
            for j, color in enumerate(uploaded_parameters["palette"]):
                red, green, blue = color["rgb"]
                color_frame = ttk.Frame(palette_frame, padding=4)
                color_frame.grid(row=j // 4, column=j % 4, padx=6, pady=4, sticky="w")
                swatch = Tk.Frame(color_frame, width=56, height=32, bg=color["hex"], relief="solid", borderwidth=1)
                swatch.pack()
                swatch.pack_propagate(False)
                ttk.Label(
                    color_frame,
                    text=f"{j + 1}: RGB=({red}, {green}, {blue})\n{color['hex']}  {color['frequency']:.1%}",
                    justify="left",
                ).pack(pady=(4, 0))

        canvas = Tk.Canvas(content_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)
        grid_frame = ttk.Frame(canvas, padding=12)
        canvas_window = canvas.create_window((0, 0), window=grid_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=1, column=0, sticky="nsew")
        scrollbar.grid(row=1, column=1, sticky="ns")

        # _event: Optional Tkinter configure event that triggers the update.
        def update_scroll_region(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        # event: Tkinter canvas resize event used to match the grid width.
        def resize_grid(event):
            canvas.itemconfigure(canvas_window, width=event.width)

        grid_frame.bind("<Configure>", update_scroll_region)
        canvas.bind("<Configure>", resize_grid)
        photos = []
        for index, texture in enumerate(textures):
            image = Image.fromarray(np.uint8(np.clip(texture, 0.0, 1.0) * 255.0))
            image.thumbnail((220, 220), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            photos.append(photo)
            card = ttk.Frame(grid_frame, padding=6)
            card.grid(row=index // 4, column=index % 4, padx=6, pady=6, sticky="nsew")
            ttk.Label(card, image=photo).pack()
            ttk.Label(card, text=f"Variation {index + 1}").pack(pady=(4, 0))
        variation_window._photos = photos

    # No parameters; generates variations using the uploaded image and extracted data.
    


    general_section = add_section("General")
    texture_type = Tk.StringVar(value="brick")
    ttk.Label(general_section, text="Texture type").pack(side="left", anchor="w")
    ttk.Combobox(general_section, textvariable=texture_type, values=("brick", "grass", "gravel"), state="readonly", width=11).pack(side="right")
    for name, default in general_defaults.items():
        add_field(general_section, name, name.replace("_", " ").title(), default)

    upload_texture_section = add_section("Upload Texture")
    ttk.Label(upload_texture_section, text="Upload texture image").pack(side="left", anchor="w")
    ttk.Button(upload_texture_section, text="Upload", command=upload_texture).pack(side="right")
    ttk.Button(
        upload_texture_section,
        text="Generate Uploaded Variations",
        command=lambda: generate_synthesized_variations(),
    ).pack(side="right", padx=(0, 8))

    brick_section = add_section("Brick Parameters")
    for name, default in brick_defaults.items():
        add_field(brick_section, name, name.replace("_", " ").title(), default)

    grass_section = add_section("GrassParameters")
    for name, default in grass_defaults.items():
        add_field(grass_section, name, name.replace("_", " ").title(), default)

    gravel_section = add_section("GravelParameters")
    for name, default in gravel_defaults.items():
        add_field(gravel_section, name, name.replace("_", " ").title(), default)

    rock_section = add_section("Rock Parameters")
    for name, default in rock_defaults.items():
        add_field(rock_section, name, name.replace("_", " ").title(), default)

    status = Tk.StringVar(value="Ready")
    generated_textures = []
    results_figure = None

    # name: Internal key for the input variable to read.
    # converter: Callable that converts the field text to the required type.
    def read_value(name, converter):
        return converter(variables[name][0].get().strip())

    # No parameters; synthesizes and displays variations from uploaded Gabor lobes.
    def generate_synthesized_variations():
        try:
            if uploaded_parameters is None or uploaded_image is None:
                raise ValueError("Upload a texture before generating variations")

            size = read_value("variation_size", int)
            image_count = read_value("image_count", int)
            seed = read_value("seed", int)
            if size < 1 or image_count < 1:
                raise ValueError("variation size and image count must be positive")

            components = uploaded_parameters["components"]
            total_energy = sum(component["energy"] for component in components) + 1e-8
            lobes = [
                {
                    "freq": component["frequency"],
                    "orientation": component["theta"],
                    "bandwidth": 1.0 / max(component["sigma_x"], component["sigma_y"], 1e-6),
                    "weight": component["energy"] / total_energy,
                }
                for component in components
            ]

            show_processing_window()
            result_queue = queue.Queue()

            def process_variations():
                try:
                    source_height, source_width = uploaded_image.shape[:2]
                    textures = []
                    for image_index in range(image_count):
                        rng = np.random.default_rng(seed + image_index)
                        scale = max(size / source_height, size / source_width)
                        resized = Image.fromarray(
                            np.uint8(uploaded_image * 255.0)
                        ).resize(
                            (
                                max(size, int(source_width * scale)),
                                max(size, int(source_height * scale)),
                            ),
                            Image.Resampling.BICUBIC,
                        )
                        max_x = resized.width - size
                        max_y = resized.height - size
                        crop_x = int(rng.integers(0, max_x + 1))
                        crop_y = int(rng.integers(0, max_y + 1))
                        crop = resized.crop((crop_x, crop_y, crop_x + size, crop_y + size))
                        if rng.random() < 0.5:
                            crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                        if rng.random() < 0.5:
                            crop = crop.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

                        base_texture = np.asarray(crop, dtype=np.float32) / 255.0
                        synthesized = normalize(
                            synthesize_variation(
                                (size, size),
                                lobes,
                                n_impulses=400,
                                seed=seed + image_index,
                            )
                        )
                        texture = np.clip(
                            base_texture + (synthesized[..., None] - 0.5) * 0.18,
                            0.0,
                            1.0,
                        )
                        texture = extractor.create_color_variation(
                            np.uint8(texture * 255.0),
                            palette=uploaded_parameters["palette"],
                        )
                        textures.append(texture.astype(np.float32) / 255.0)
                    result_queue.put((textures, None))
                except Exception as error:
                    result_queue.put((None, error))

            def finish_variations():
                try:
                    textures, error = result_queue.get_nowait()
                except queue.Empty:
                    root.after(100, finish_variations)
                    return

                hide_processing_window()
                if error is not None:
                    messagebox.showerror("Variation generation failed", str(error), parent=root)
                    status.set("Variation generation failed")
                    return

                generated_textures[:] = textures
                show_variations(textures)
                status.set(f"Generated {image_count} synthesized variation(s) at {size}x{size}")

            threading.Thread(target=process_variations, daemon=True).start()
            root.after(100, finish_variations)
        except (TypeError, ValueError) as error:
            messagebox.showerror("Variation generation failed", str(error), parent=root)
            status.set("Variation generation failed")

    # No parameters; exports the currently generated textures to a CSV file.
    def export_textures():
        if not generated_textures:
            messagebox.showinfo("No results", "Generate at least one texture first.", parent=root)
            return

        file_path = filedialog.asksaveasfilename(
            title="Export texture data",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not file_path:
            return

        with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(("image", "row", "column", "intensity"))
            for image_index, texture in enumerate(generated_textures, start=1):
                for row, column in np.ndindex(texture.shape):
                    value = texture[row, column]
                    if np.ndim(value) > 0:
                        value = np.mean(value)
                    writer.writerow((image_index, row, column, float(value)))
        status.set(f"Exported {len(generated_textures)} texture(s) to CSV")

    # No parameters; generates textures from the values currently entered in the form.
    def generate_from_form():
        nonlocal results_figure
        try:
            height = read_value("height", int)
            width = read_value("width", int)
            image_count = read_value("image_count", int)
            texture_rotation = read_value("texture_rotation", float)
            seed = read_value("seed", int)
            if height < 1 or width < 1 or image_count < 1:
                raise ValueError("height, width, and image count must be positive")

            selected_type = texture_type.get()
            textures = []
            for image_index in range(image_count):
                image_seed = seed + image_index
                if selected_type == "brick":
                    parameters = {name: read_value(name, float if isinstance(default, float) else int) for name, default in brick_defaults.items()}
                    parameters["theta"] = np.deg2rad(parameters["theta"])
                    parameters["vertical_theta"] = np.deg2rad(parameters["vertical_theta"])
                    parameters["mortar_theta"] = np.deg2rad(parameters["mortar_theta"])
                    texture = make_brick_texture(height=height, width=width, texture_rotation=texture_rotation, seed=image_seed, **parameters)

                elif selected_type == "grass":
                    parameters = {name: read_value(name, float) for name in grass_defaults}
                    texture = make_grass_texture(
                        height=height,
                        width=width,
                        frequency=parameters["frequency"],
                        theta=np.deg2rad(parameters["theta"] + texture_rotation),
                        sigma_x=parameters["sigma_x"],
                        sigma_y=parameters["sigma_y"],
                        seed=image_seed,
                    )

                elif selected_type == "gravel":
                
                    parameters = {name: read_value(name, float) for name in gravel_defaults}
                    texture = make_gravel_texture(
                        height=height,
                        width=width,
                        frequency=parameters["frequency"],
                        theta=np.deg2rad(parameters["theta"] + texture_rotation),
                        sigma_x=parameters["sigma_x"],
                        sigma_y=parameters["sigma_y"],
                        seed=image_seed,
                    )
                elif selected_type == "rock":
                    parameters = {name: read_value(name, float) for name in rock_defaults}
                    texture = make_rock_texture(
                        height=height,
                        width=width,
                        frequency=parameters["frequency"],
                        theta=np.deg2rad(parameters["theta"] + texture_rotation),
                        sigma_x=parameters["sigma_x"],
                        sigma_y=parameters["sigma_y"],
                        seed=image_seed,
                    )
                textures.append(texture)
            generated_textures[:] = textures
            if results_figure is not None:
                plt.close(results_figure)
            columns = min(4, max(1, image_count))
            rows = int(np.ceil(image_count / columns))
            results_figure, axes = plt.subplots(rows, columns, squeeze=False, figsize=(4 * columns, 4 * rows))
            for image_index, texture in enumerate(textures):
                axis = axes.flat[image_index]
                axis.imshow(texture, cmap="gray", vmin=0, vmax=1)
                axis.set_title(f"{selected_type} {image_index + 1}")
                axis.axis("off")
            for axis in axes.flat[image_count:]:
                axis.axis("off")
            results_figure.tight_layout()
            results_figure.show()
            status.set(f"Generated {image_count} {selected_type} texture(s)")
        except (TypeError, ValueError) as error:
            messagebox.showerror("Invalid input", str(error), parent=root)
            status.set("Invalid input")

    ttk.Button(form_frame, text="Generate Textures", command=generate_from_form).pack(anchor="e", pady=(0, 8))
    ttk.Button(form_frame, text="Export Results as CSV", command=export_textures).pack(anchor="e", pady=(0, 8))
    ttk.Label(form_frame, textvariable=status).pack(anchor="w")
    root.mainloop()
    


# image: Numeric array whose values should be rescaled to the range 0.0 to 1.0.
def normalize(image):
    image = image - image.min()
    return image / (image.max() + 1e-8)




# height: Number of rows in the generated noise image.
# width: Number of columns in the generated noise image.
# frequency: Gabor carrier frequency.
# theta: Gabor orientation in radians.
# sigma_x: Gaussian spread along the kernel x axis.
# sigma_y: Gaussian spread along the kernel y axis.
# seed: Random seed for reproducible source noise.
def make_gabor_noise(
    height=512,
    width=512,
    frequency=0.08,
    theta=0.0,
    sigma_x=5.0,
    sigma_y=2.0,
    seed=42,
):
    rng = np.random.default_rng(seed)

    # Random source signal
    random_signal = rng.normal(0, 1, (height, width))

    # Gabor kernel contains real and imaginary components.
    kernel = gabor_kernel(
        frequency=frequency,
        theta=theta,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
    )

    real_response = ndimage.convolve(
        random_signal,
        np.real(kernel),
        mode="wrap",
    )

    imaginary_response = ndimage.convolve(
        random_signal,
        np.imag(kernel),
        mode="wrap",
    )

    # Magnitude makes the noise independent of phase.
    noise = np.sqrt(real_response**2 + imaginary_response**2)

    noise -= noise.mean()
    noise /= noise.std() + 1e-8

    return noise

#|||||||||||||||||||||||||||||||||||||||||||| Deprecated the texture specifics 
# height: Number of rows in the generated texture.
# width: Number of columns in the generated texture.
# brick_width: Width of one brick in pixels.
# brick_height: Height of one brick in pixels.
# mortar_width: Width of vertical mortar in pixels.
# mortar_height: Height of horizontal mortar in pixels.
# texture_rotation: Rotation of the brick texture in degrees.
# frequency: Horizontal Gabor frequency.
# theta: Horizontal Gabor orientation in radians.
# sigma_x: Horizontal Gabor x spread.
# sigma_y: Horizontal Gabor y spread.
# vertical_frequency: Vertical Gabor frequency.
# vertical_theta: Vertical Gabor orientation in radians.
# vertical_sigma_x: Vertical Gabor x spread.
# vertical_sigma_y: Vertical Gabor y spread.
# horizontal_weight: Contribution of horizontal noise.
# vertical_weight: Contribution of vertical noise.
# mortar_frequency: Mortar Gabor frequency.
# mortar_theta: Mortar Gabor orientation in radians.
# mortar_sigma_x: Mortar Gabor x spread.
# mortar_sigma_y: Mortar Gabor y spread.
# brick_base: Base grayscale value for bricks.
# brick_contrast: Variation strength applied to bricks.
# mortar_base: Base grayscale value for mortar.
# mortar_contrast: Variation strength applied to mortar.
# seed: Random seed used for reproducible noise.
def make_brick_texture(
    height,
    width,
    brick_width,
    brick_height,
    mortar_width,
    mortar_height,
    texture_rotation,
    frequency,
    theta,
    sigma_x,
    sigma_y,
    vertical_frequency,
    vertical_theta,
    vertical_sigma_x,
    vertical_sigma_y,
    horizontal_weight,
    vertical_weight,
    mortar_frequency,
    mortar_theta,
    mortar_sigma_x,
    mortar_sigma_y,
    brick_base,
    brick_contrast,
    mortar_base,
    mortar_contrast,
    seed,

):
    y, x = np.indices((height, width))

    row = y // brick_height
    row_offset = (row % 2) * (brick_width / 2)

    local_x = (x + row_offset) % brick_width
    local_y = y % brick_height
    texture_angle = np.deg2rad(texture_rotation)

    # True where the pixel belongs to mortar.
    mortar = (
        (local_x < mortar_width)
        | (local_y < mortar_height)
    )

    # Gabor noise at two orientations.
    horizontal_noise = make_gabor_noise(
        height,
        width,
        frequency=frequency,
        theta=theta + texture_angle,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        seed=seed,
    )

    vertical_noise = make_gabor_noise(
        height,
        width,
        frequency=vertical_frequency,
        theta=vertical_theta + texture_angle,
        sigma_x=vertical_sigma_x,
        sigma_y=vertical_sigma_y,
        seed=seed + 1,
    )

    # Combine directional components.
    surface_noise = (
        horizontal_weight * horizontal_noise
        + vertical_weight * vertical_noise
    )

    surface_noise = normalize(surface_noise)

    # Base brick color and variation.
    brick = brick_base + brick_contrast * surface_noise

    # Darker mortar with a little variation.
    mortar_noise = make_gabor_noise(
        height,
        width,
        frequency=mortar_frequency,
        theta=mortar_theta + texture_angle,
        sigma_x=mortar_sigma_x,
        sigma_y=mortar_sigma_y,
        seed=seed + 2,
    )

    mortar_texture = mortar_base + mortar_contrast * normalize(mortar_noise)

    texture = np.where(mortar, mortar_texture, brick)

    return np.clip(texture, 0.0, 1.0)

#gotta differentiate between calculations of grass and gravel here onwards womp
# height: Number of rows in the generated texture.
# width: Number of columns in the generated texture.
# frequency: Strand density and broad-noise frequency.
# theta: Strand orientation in radians.
# sigma_x: Strand spread along the x axis.
# sigma_y: Strand spread along the y axis.
# seed: Random seed for reproducible grass noise.
def make_grass_texture(
    height,
    width,
    frequency,
    theta,
    sigma_x,
    sigma_y,
    seed,
):
    rng = np.random.default_rng(seed)
    density = np.clip(frequency * 1.8, 0.035, 0.18)
    impulse_field = (rng.random((height, width)) < density).astype(np.float32)

    # Blur sparse impulses into elongated strands in the requested direction.
    aligned = ndimage.rotate(
        impulse_field,
        angle=-np.rad2deg(theta),
        reshape=False,
        order=1,
        mode="reflect",
    )
    strands = ndimage.gaussian_filter(
        aligned,
        sigma=(max(sigma_y, 0.5), max(sigma_x, 0.5)),
        mode="wrap",
    )
    strands = ndimage.rotate(
        strands,
        angle=np.rad2deg(theta),
        reshape=False,
        order=1,
        mode="reflect",
    )
    strands = normalize(strands)

    broad_variation = make_gabor_noise(
        height=height,
        width=width,
        frequency=max(frequency * 0.45, 0.015),
        theta=theta,
        sigma_x=max(sigma_x * 1.5, 2.0),
        sigma_y=max(sigma_y, 1.0),
        seed=seed + 1,
    )
    broad_variation = normalize(broad_variation)
    grass_texture = 0.12 + 0.68 * strands + 0.20 * broad_variation
    return np.clip(grass_texture, 0.0, 1.0)

# height: Number of rows in the generated texture.
# width: Number of columns in the generated texture.
# frequency: Gabor carrier frequency for gravel noise.
# theta: Gabor orientation in radians.
# sigma_x: Gaussian spread along the kernel x axis.
# sigma_y: Gaussian spread along the kernel y axis.
# seed: Random seed for reproducible noise.
def make_gravel_texture(
    height,
    width,
    frequency,
    theta,
    sigma_x,
    sigma_y,
    seed,
):
    gravel_noise = make_gabor_noise(
        height=height,
        width=width,
        frequency=frequency,
        theta=theta,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        seed=seed,
    )

    gravel_texture = normalize(gravel_noise)

    return gravel_texture


# height: Number of rows in the generated texture.
# width: Number of columns in the generated texture.
# frequency: Gabor carrier frequency for rock noise.
# theta: Gabor orientation in radians.
# sigma_x: Gaussian spread along the kernel x axis.
# sigma_y: Gaussian spread along the kernel y axis.
# seed: Random seed for reproducible noise.
def make_rock_texture(
    height,
    width,
    frequency,
    theta,
    sigma_x,
    sigma_y,
    seed,
):
    rock_noise = make_gabor_noise(
        height=height,
        width=width,
        frequency=frequency,
        theta=theta,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        seed=seed,
    )

    rock_texture = normalize(rock_noise)

    return rock_texture


"""
To create variation from image, and using gabor's dependency on power spectrum...
"""

# n_lobes: Maximum number of dominant power-spectrum peaks to return.
def find_lobes(n_lobes=6, power_spectrum=None):
    # Find local maxima in the power spectrum -> candidate (freq, orientation) pairs
    if power_spectrum is None:
        power_spectrum = extractor.get_power_spectrum()
    
    local_max = ndimage.maximum_filter(power_spectrum, size=9) == power_spectrum
    peaks = np.argwhere(local_max & (power_spectrum > power_spectrum.mean() * 5))
    peaks = sorted(peaks, key=lambda p: -power_spectrum[tuple(p)])[:n_lobes]
    return peaks  # each peak -> (fx, fy) gives frequency + orientation, local spread gives bandwidth


# No parameters; placeholder for future kernel-generation logic.
def kernel(x, y, frequency, theta, bandwidth):
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    gaussian = np.exp(-np.pi * bandwidth**2 * (xr**2 + yr**2))
    return gaussian * np.cos(2 * np.pi * frequency * xr)



def extract_bandwidth(gray_img, power_spectrum):
    img = gray_img.astype(np.float64)
    img -= img.mean()
    H, W = power_spectrum.shape
    cy, cx = H // 2, W // 2
    power_spectrum[cy-2:cy+3, cx-2:cx+3] = 0  # zero out the DC spike

    # locate the dominant frequency peak
    py, px = np.unravel_index(np.argmax(power_spectrum), power_spectrum.shape)
    f0 = np.hypot(px - cx, py - cy) / max(H, W)  # cycles/pixel

    # fit a 2D Gaussian around the peak to get its spread (sigma_f)
    r = 10
    y, x = np.mgrid[py-r:py+r, px-r:px+r]
    patch = power_spectrum[py-r:py+r, px-r:px+r]


    def gauss2d(coords, amp, sigma):
        yy, xx = coords
        return amp * np.exp(-((xx - px)**2 + (yy - py)**2) / (2 * sigma**2))

    (amp, sigma_f_px), _ = curve_fit(gauss2d, np.vstack([y.ravel(), x.ravel()]),
                                      patch.ravel(), p0=[patch.max(), 3])
    sigma_f = sigma_f_px / max(H, W)  # normalize to cycles/pixel

    # convert spatial-frequency spread -> octave bandwidth
    sigma_spatial = 1.0 / (2 * np.pi * sigma_f)
    k = sigma_spatial * f0 * np.pi / np.sqrt(np.log(2) / 2)
    bandwidth = np.log2((k + 1) / (k - 1))
    return f0, bandwidth

def synthesize_variation(shape, lobes, n_impulses=4000, seed=None):
    rng = np.random.default_rng(seed)
    height, width = shape
    out = np.zeros(shape)

    xs, ys = np.meshgrid(np.arange(width) - width/2, np.arange(height) - height/2)
    for lobe in lobes:
        n = int(n_impulses * lobe["weight"])
        for _ in range(n):
            cx, cy = rng.uniform(-width/2, width/2), rng.uniform(-height/2, height/2)
            phase = rng.uniform(0, 2*np.pi)
            ori = lobe["orientation"] + rng.normal(0, 0.05)  # jitter
            out += kernel(xs - cx, ys - cy, lobe["freq"], ori, lobe["bandwidth"]) * np.cos(phase)
    return out


if __name__ == "__main__":
    userInputMode()

    