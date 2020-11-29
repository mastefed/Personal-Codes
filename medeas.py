import tkinter
from PIL import Image, ImageTk

def import_image():
    imported_image = tkinter.Tk()
    imported_image.title('Immagine Importata')

    img = Image.open(image_path.get())
    image = ImageTk.PhotoImage(img, master=imported_image)
    width, height = img.size

    canvas = tkinter.Canvas(imported_image, width=width, height=height)
    canvas.pack()

    canvas.create_image(width/2,height/2,image=image)
    imported_image.mainloop()
    pass


root = tkinter.Tk()
root.geometry('400x150')
root.title('MedE(a)s')
root.resizable(False,False)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

welcome = tkinter.Label(
    root,
    text = "Importa l'immagine da analizzare:",
    font=('Noto, 10'),
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
    column=1,
    pady=20
)

download_button = tkinter.Button(
    text="Importa",
    command=import_image
)
download_button.grid(
    row=1,
    column=1
)

if __name__ == "__main__":
    root.mainloop()