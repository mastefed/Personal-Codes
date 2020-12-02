import tkinter
from PIL import Image, ImageTk

# /home/fvm/Immagini/ff.jpg

def import_image():
    imported_image = tkinter.Tk()
    imported_image.title('Imported Image')

    transformations = tkinter.Tk()
    transformations.title('Transformations')
    transformations.configure(bg='white')
    transformations.columnconfigure(0, weight=1)
    transformations.columnconfigure(1, weight=1)

    img = Image.open(image_path.get())

    rotation_lbl = tkinter.Label(
        master=transformations,
        text='Choose the rotation degree:',
        bg='white',
        fg='black'
    )
    rotation_lbl.grid(
        row=0,
        column=0,
        padx=10,
        pady=10
    )

    rotation_entry = tkinter.Entry(
        master=transformations,
        width=5,
        background='white',
        foreground='black'
    )
    rotation_entry.insert(tkinter.END, '0.0')
    rotation_entry.grid(
        row=0,
        column=1,
        padx=10
    )

    rotation_button = tkinter.Button(
        master=transformations,
        text="Rotate",
        command = img.rotate(float(rotation_entry.get())),
        bg='white',
        fg='blue'
    )
    rotation_button.grid(
        row=0,
        column=2
    )

    image = ImageTk.PhotoImage(img, master=imported_image)
    width, height = img.size

    canvas = tkinter.Canvas(imported_image, width=width, height=height)
    canvas.pack()

    canvas.create_image(width/2,height/2,image=image)

    imported_image.mainloop()
    transformations.mainloop()
    pass


root = tkinter.Tk()
root.geometry('500x150')
root.title('MedE(a)s')
root.resizable(False,False)
root.configure(bg='white')
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=1)

welcome = tkinter.Label(
    root,
    text = "Import an image to show and analyze it:",
    font=('Noto, 10'),
    bg='white',
    fg='black'
)
welcome.grid(
    row=0,
    column=0,
    pady=20
)

image_path = tkinter.Entry(
    background='white',
    foreground='black'
)
image_path.grid(
    row=0,
    column=1
)

upload_button = tkinter.Button(
    text="Import",
    command=import_image,
    bg='white',
    fg='blue'
)
upload_button.grid(
    row=0,
    column=2
)

if __name__ == "__main__":
    root.mainloop()