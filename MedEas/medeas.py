import tkinter
from PIL import Image, ImageTk

def import_image():
    imported_image = tkinter.Tk()
    imported_image.title('Imported Image')

    transformations = tkinter.Tk()
    transformations.title('Transformations')
    transformations.configure(bg='white')
    transformations.columnconfigure(0, weight=1)

    img = Image.open(image_path.get())
    image = ImageTk.PhotoImage(img, master=imported_image)
    width, height = img.size

    canvas = tkinter.Canvas(imported_image, width=width, height=height)
    canvas.pack()

    canvas.create_image(width/2,height/2,image=image)

    imported_image.mainloop()
    transformations.mainloop()
    pass


root = tkinter.Tk()
root.geometry('450x150')
root.title('MedE(a)s')
root.resizable(False,False)
root.configure(bg='white')
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=1)

welcome = tkinter.Label(
    root,
    text = "Importa l'immagine da analizzare:",
    font=('Noto, 10'),
    bg='white',
    fg='black'
)
welcome.grid(
    row=0,
    column=0,
    sticky='n', pady=20
)

image_path = tkinter.Entry(
    background='white',
    foreground='black'
)
image_path.grid(
    row=0,
    column=1
)

download_button = tkinter.Button(
    text="Importa",
    command=import_image,
    bg='white',
    fg='blue'
)
download_button.grid(
    row=0,
    column=2
)

if __name__ == "__main__":
    root.mainloop()