from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

root=Path(__file__).resolve().parents[1]
files=sorted((root/'figures').glob('0*.png'))
canvas=Image.new('RGB',(1600,2000),'white')
draw=ImageDraw.Draw(canvas)
for index,path in enumerate(files):
    picture=Image.open(path).convert('RGB')
    picture.thumbnail((780,465))
    left=(index%2)*800+(800-picture.width)//2
    top=(index//2)*500+25
    canvas.paste(picture,(left,top))
    draw.text(((index%2)*800+12,(index//2)*500+5),path.stem,fill='black')
canvas.save(root/'figures/contact_sheet.png')
