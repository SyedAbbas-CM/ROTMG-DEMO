//import map from "./mapd.js";
const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d", { alpha: false });
ctx.imageSmoothingEnabled = false;
ctx.webkitImageSmoothingEnabled = false;

const gameCanvas = new OffscreenCanvas(600,600)
const gctx = gameCanvas.getContext("2d", { alpha: false });
gctx.imageSmoothingEnabled = false;
gctx.webkitImageSmoothingEnabled = false;

const c3w = 128
const canvas3 = new OffscreenCanvas(c3w,c3w)
const ctx3 = canvas3.getContext("2d")

ctx3.imageSmoothingEnabled = false;
ctx3.webkitImageSmoothingEnabled = false;

const envi = new Image();
envi.src = "assets/images//lofiEnvironment.png";
const char = new Image();
char.src = "assets/images/lofiChar.png";
const play = new Image();
play.src = "sprites/players.png";
const env2 = new Image();
env2.src = "sprites/lofiEnvironment2.png";
//bullets and misc objects
const obj4 = new Image();
obj4.src = "sprites/lofiObj.png";

const agj = new FontFace("agj", "url(adoquin/adoquin.woff2)");
agj.load().then((font) => {
  document.fonts.add(font);
  ctx.font = "25px agj"
  gctx.font = "25px agj"
});

let namae = prompt("name?",0)

const p = { name: namae, x: 50, y: 50, r: 0.003, tx: 0, ty: 0 }; // r:0 causes walls to not be drawn



const tSize = 50,
  tHalf = tSize / 2

const screen = 600,
  xCenter = 300;
let yCenter = 324;

let offCenter = false
const offCenterDiff = 450 - 324

const min = -tSize * 1.5,
  max = screen + tSize * 2;


const scanC1 = -9,
  scanC2 = 11;
const scanOC1 = -11,
  scanOC2 = 13;
let scan1 = scanC1,
  scan2 = scanC2;

const WALL_SIZE = 1

let speed;
let standardSpeed = 0.075;
let diagonalSpeed = 0.707 * standardSpeed;

const wallEdge = [0, 0, tSize, 0, 0, 0, 0, tSize, tSize, 0, tSize, tSize, 0, tSize, tSize, tSize];
let lowestPoint = [0, 0, 0, 0];


let wallWidth = [];
let wallStartX = [];
let wallStartY = [];
let wallEndX = [];
let wallEndY = [];
let wallRatio = [];

let enemies = []

function gameLoop() {
  
  if(!draw_shit && !create_object && !picking_texture){
    
  if (key.E) p.r -= 0.0233;
  if (key.Q) p.r += 0.0233;

  const sin = Math.sin(p.r);
  const cos = Math.cos(p.r);

  if (key.W + key.A + key.S + key.D > 1) speed = diagonalSpeed;
  else speed = standardSpeed;

  if (key.W) {
    let dx = Math.cos(-1.5708 - p.r);
    let dy = Math.sin(-1.5708 - p.r);
    mapCollision(dx, dy);
  }
  if (key.S) {
    let dx = Math.cos(1.5708 - p.r);
    let dy = Math.sin(1.5708 - p.r);
    mapCollision(dx, dy);
  }
  if (key.D) mapCollision(cos, -sin);
  if (key.A) mapCollision(-cos, sin);

  let xDiff = Math.floor(p.x) - p.x;
  let yDiff = Math.floor(p.y) - p.y;

  let entity = [
    {
      tex: char,
      tx: p.tx,
      ty: p.ty,
      x: xCenter,
      y: yCenter,
      size: 40,
      wall:false
    },
  ]

  let nearPX1 = p.x - 15
  let nearPX2 = p.x + 15
  let nearPY1 = p.y - 15
  let nearPY2 = p.y + 15

  enemies.forEach(e => {
    if(e.x > nearPX1 &&
      e.x < nearPX2 &&
      e.y > nearPY1 &&
      e.y < nearPY2
    ){

      let offCenterPlus = offCenter ? offCenterDiff : 0
      let screenX = (tSize * (e.x - p.x + 5)) -xCenter
      let screenY = (tSize * (e.y - p.y + 5)) - yCenter + offCenterPlus

      let rotatedScreenX = screenX * cos + screenY * -sin + xCenter
      let rotatedScreenY = screenX * sin + screenY * cos + yCenter

      entity.push({
        ...e,
        x: rotatedScreenX,
        y: rotatedScreenY
      })}

  })

  let roofs = []  

  gctx.save();
  gctx.translate(xCenter, yCenter);
  gctx.rotate(p.r);
  for (let x = scan1; x < scan2; x++) {
    for (let y = scan1; y < scan2; y++) {

      let outOfBounds = p.x + x < 0 ||
                        p.x + x > mapSize ||
                        p.y + y < 0 ||
                        p.y + y > mapSize

      let tileX = (tSize * (xDiff + x)) -tHalf
      let tileY = (tSize * (yDiff + y)) -tHalf

      // always top left corner of tile
      let rotatedTileX = tileX * cos + tileY * -sin + xCenter
      let rotatedTileY = tileX * sin + tileY * cos + yCenter

      if (
        rotatedTileX > min &&
        rotatedTileX < max &&
        rotatedTileY > min &&
        rotatedTileY < max
      ) {
        // floor because decimal values get multiplied into incorrect map location
        let arrayLocation = Math.floor(p.x) + x + mapSize * (Math.floor(p.y) + y);
        let texture = map[arrayLocation];
        let tileData = texMap.get(texture);

        if (tileData === undefined || outOfBounds)
            {
            tileData = { x: 48, y: 48, tex: envi, wall: false, solid:false, obstacle:false, deco: false};
            }

        gctx.drawImage(tileData.tex,tileData.x,tileData.y,8,8,tileX,tileY,tSize,tSize);

        if(tileData.deco) gctx.drawImage(tileData.tex,tileData.x2,tileData.y2,8,8,tileX,tileY,tSize,tSize);

        if(tileData.obstacle){
          let obstX = tSize * (xDiff + x)
          let obstY = tSize * (yDiff + y)
    
          // always top left corner of tile
          let obstacleX = Math.round(obstX * cos + obstY * -sin + xCenter)
          let obstacleY = Math.round(obstX * sin + obstY * cos + yCenter)

          entity.push({
            tex: tileData.tex,
            tx: tileData.x2,
            ty: tileData.y2,
            x:obstacleX,
            y:obstacleY,
            size:tileData.size
        })
      }

        if (tileData.wall) {
          roofs.push({tex:tileData.tex,tx:tileData.x,ty:tileData.y,x:tileX,y:tileY,});

          for (let n = 0; n < 4; n++) {
            let n4 = n * 4;
            let edgeX = tileX + wallEdge[n4 + 0];
            let edgeY = tileY + wallEdge[n4 + 1];
            let rotatedEdgeX = edgeX * cos + edgeY * -sin;
            let rotatedEdgeY = edgeX * sin + edgeY * cos;
            edgeX = tileX + wallEdge[n4 + 2];
            edgeY = tileY + wallEdge[n4 + 3];
            let rotatedEdgeX2 = edgeX * cos + edgeY * -sin;
            let rotatedEdgeY2 = edgeX * sin + edgeY * cos;

            lowestPoint[n] = rotatedEdgeY > rotatedEdgeY2 ? rotatedEdgeY : rotatedEdgeY2; // find which walls to draw
            wallWidth[n] = Math.abs(rotatedEdgeX - rotatedEdgeX2); // find how many lines are needed
     
            wallStartX[n] = Math.round(rotatedEdgeX + xCenter)
            wallStartY[n] = rotatedEdgeY + yCenter

            wallEndX[n] = rotatedEdgeX2 + xCenter
            wallEndY[n] = rotatedEdgeY2 + yCenter

            wallRatio[n] = (rotatedEdgeY - rotatedEdgeY2) / (rotatedEdgeX - rotatedEdgeX2)

          }

          let rank = rankArray(lowestPoint); //no walls on 0 degree problem

            for (let g = 0; g < 4; g++) { 
              if (rank[g] < 2 && texMap.has(map[arrayLocation + adjacentNWES[g]]) && !texMap.get(map[arrayLocation + adjacentNWES[g]]).wall){// if undefined adjacant = carsh

                let sign = wallStartX[g] - wallEndX[g] < 0 ? WALL_SIZE : -WALL_SIZE
                let end = wallWidth[g]
                let inverseWidthBy8 = (1 / wallWidth[g]) * 8

                for (let z = 0; Math.abs(z) < end; z+= sign){

                  entity.push({
                    tex: tileData.tex,
                    tx: tileData.x2 + Math.floor(Math.abs(z) * inverseWidthBy8),
                    ty: tileData.y2,

                    x: wallStartX[g] + z,
                    y: Math.round(wallStartY[g] + z * wallRatio[g]),

                    wall: true,
                  });
                  
                  }
                }   
              } 

        } 
      }
    }
  }
  gctx.restore();

  entity.sort((a, b) => a.y - b.y);

  entity.forEach(e => {
    if (e.wall)
      gctx.drawImage(e.tex, e.tx, e.ty, 1, 8, e.x, e.y, WALL_SIZE, -tSize);
    else
      {
        
        gctx.drawImage(e.tex,e.tx,e.ty,8,8,Math.round(e.x) - e.size * 0.5,Math.round(e.y) - e.size,e.size,e.size);
        if(e.name && e.name !== namae){
          gctx.fillStyle = "rgb(255,255,255)"
          gctx.fillText(e.name,e.x - 20,e.y + 20)
        }
      }
  });

  if(roofs.length){
    gctx.save();
    gctx.translate(xCenter, yCenter - tSize );
    gctx.rotate(p.r);
    roofs.forEach( e =>{gctx.drawImage(e.tex, e.tx, e.ty, 8, 8, e.x - 1, e.y - 1, tSize+2, tSize+2);})
    gctx.restore();
  }

  tinf = texMap.get(buildingBlock)

  if(tinf.hasOwnProperty('x2'))  gctx.drawImage(tinf.tex, tinf.x2, tinf.y2, 8, 8, 600 - 50 , tSize, tSize, tSize);
  gctx.drawImage(tinf.tex, tinf.x, tinf.y, 8, 8, 600 - 50 , 0, tSize, tSize);

  if(key[" "]) {
    let mapLocation = (Math.round(p.x - sin)) + (mapSize * (Math.round(p.y - cos)))
    if(map[mapLocation] !== buildingBlock){
    map[mapLocation] = buildingBlock
    socket.send(JSON.stringify({ type: 'MAP_CHANGE', location: mapLocation, block: buildingBlock }));
    }
  }


  } // END OF GAME

  if(draw_shit && !create_object && !picking_texture){// PRESS J
    gctx.clearRect(0,0,canvas.width,canvas.height)
    gctx.fillStyle = "rgb(20,20,20)"
    gctx.fillRect(0,0,canvas.width,canvas.height)

    let c3MouseX = Math.floor(mouseX / zoom)
    let c3MouseY = Math.floor(mouseY / zoom)
    
    if(color_picker && mouseDown){
      [r,g,b] = getPixelColor(c3MouseX - Math.floor(drawing_x /zoom),c3MouseY - Math.floor(drawing_y / zoom))
      color_picker = false
    }

    gctx.drawImage(canvas3,drawing_x,drawing_y,Math.round(c3w*zoom), Math.round(c3w*zoom)) // MAIN DRAWING CANVAS
    gctx.fillStyle= "black"
    let stripes = (c3w / 8) + 1
    for (let x = 0; x < stripes; x++) {
      for (let y = 0; y < stripes; y++) {
        gctx.fillRect(drawing_x ,drawing_y + (y * zoom * 8),c3w * zoom,1)
        gctx.fillRect(drawing_x + (x * zoom * 8),drawing_y + y,1,c3w * zoom)
      }
    }
    
     
    gctx.drawImage(canvas3,screen -c3w,screen - c3w,Math.round(c3w), Math.round(c3w)) // smol spriteshet


    // CAM MOVEMENT
    if(within(screen-c3w,screen-c3w,screen,screen)){
      if(mouseX < screen -c3w + c3w / 2) drawing_x += 3
      else drawing_x -= 3
      if(mouseY < screen -c3w + c3w / 2) drawing_y += 3
      else drawing_y -= 3
    }




    gctx.fillStyle = "black"
    gctx.fillRect(539,0,60,256)

    gctx.fillStyle = rgb(`${r},${g},${b}`)
    gctx.fillRect(540,256,60,60)
    
    gctx.fillStyle = "white"
    gctx.fillText(`${r},${g},${b}`,490,340)


    if(button("pick",525,350,70,24) && !erasing) color_picker = true

    if(button("erase",525,380,70,24) && !color_picker && eraseDelay){
      eraseDelay = false
      erasing = !erasing

      setTimeout(() => {eraseDelay = true}, 100);
    }

    gctx.fillStyle = rgb(`${r}`,0,0)
    gctx.fillRect(540,0,20,r)
    gctx.fillStyle = rgb(0,`${g}`,0)
    gctx.fillRect(560,0,20,g)
    gctx.fillStyle = rgb(0,0,`${b}`)
    gctx.fillRect(580,0,20,b)


    if(within(540,-1,20,256,2)) r = mouseY
    if(within(560,-1,20,256,2)) g = mouseY
    if(within(580,-1,20,256,2)) b = mouseY


    ctx3.fillStyle = rgb(`${r},${g},${b}`)

    for (let n = 0; n < last_used_colors.length; n++) {
      gctx.fillStyle = last_used_colors[n]
      gctx.fillRect(n * 20,580,20,20)
    }

    if(within(0,580,last_used_colors.length * 20,20,2)){
        let rgbstring = last_used_colors[Math.floor(mouseX / 20)]
        ctx3.fillStyle = rgbstring
 
       const rgbArray = rgbstring.match(/\d+/g).map(Number);
       [r, g, b] = rgbArray;
    }

    if(within(0,0,c3w,c3w,3,c3MouseX,c3MouseY)  && !erasing && mouseX < 525){

      ctx3.fillRect(c3MouseX - Math.floor(drawing_x / zoom),c3MouseY - Math.floor(drawing_y / zoom),1,1)

      if (!last_used_colors.includes(rgb(`${r},${g},${b}`))) {
        last_used_colors.push(rgb(`${r},${g},${b}`));
      }
      
    }
      if(mouseDown && erasing) ctx3.clearRect(c3MouseX  - Math.floor(drawing_x / zoom),c3MouseY - Math.floor(drawing_y / zoom),1,1)
  }

  if(!draw_shit && create_object){
    
    if(!picking_texture){

        gctx.fillStyle = "rgb(20,20,20)"
        gctx.fillRect(0,0,screen,screen)

      if(button("New tile",210,165,180,24)){
        object_options.deco = true
        picking_texture = true
        setTimeout(() => {button_timeout = true}, 500);
      }

      if(button("New block",210,195,180,24)){
        object_options.wall = true
        object_options.solid = true
        picking_texture = true
        setTimeout(() => {button_timeout = true}, 500);
      }

      if(button("New obstacle",210,225,180,24)){
        object_options.obstacle = true
        object_options.solid = true
        picking_texture = true
        setTimeout(() => {button_timeout = true}, 500);
      }

      if(button("New decoration",210,255,180,24)){
        object_options.obstacle = true
        picking_texture = true
        setTimeout(() => {button_timeout = true}, 500);
      }
    }

    if(picking_texture){
      gctx.fillStyle = "rgb(20,20,20)"
      gctx.fillRect(0,0,screen,screen)

      gctx.fillStyle = "rgb(30,30,30)"
      gctx.fillRect(200,550,48,48)
      gctx.fillRect(260,550,48,48)

      gctx.drawImage(canvas3,object_options.x,object_options.y,8, 8,205,555,40,40)
      gctx.drawImage(canvas3,object_options.x2,object_options.y2,8, 8,265,555,40,40)

      gctx.drawImage(canvas3,0,0,c3w * 4, c3w* 4)



      gctx.fillStyle = "rgb(255,255,255)"
      gctx.fillText("pick two textures",10,570)
      gctx.fillText("#1: tile #2: misc",10,590)

      for (let x = 0; x < 16; x++) {
        for (let y = 0; y < 16; y++) {
          if(mouseX > (x * 32) && mouseX < (x * 32) + 32 && mouseY > (y * 32) && mouseY < (y * 32) + 32){
            gctx.fillRect(x*32,y*32,32,32)
            if(mouseDown && button_timeout){
              if(texture_swap_flag){
                object_options.x = x * 8
                object_options.y = y * 8
                texture_swap_flag = false
                button_timeout = false
                setTimeout(() => {button_timeout = true}, 300);
              }
              else{
                object_options.x2 = x * 8 
                object_options.y2 = y * 8
                texture_swap_flag = true
                button_timeout = false
                setTimeout(() => {button_timeout = true}, 300);
              }
            }}
        }
      }

      if(button("finish",520,570,76,24)){

/*         
        texMap.set(tex_map_size, object_options);
        tex_map_size++  */

        object_options.tex = "canvas3"
        socket.send(JSON.stringify({ type: 'NEW_TEXTURE_MAP', options: object_options }))
        object_options.tex = canvas3

        picking_texture = false
        create_object = false



        object_options = { 
          x: 0,
          y: 0,
          tex: canvas3,
          wall: false,
          obstacle: false,
          x2: 0,
          y2: 0,
          size: 50,
          solid: false,
          deco: false
         }


      }


    }
  }

  ctx.drawImage(gameCanvas,0,0)

  updateBullets();
  renderBullets();
  requestAnimationFrame(gameLoop);
}

function shootBullet() {
  // Determine bullet's initial position and direction based on player's direction
  const bulletSpeed = 1; // Example speed, adjust as needed
  const bullet = {
    x: p.x,
    y: p.y,
    direction: p.r, // Use player's current direction
    speed: bulletSpeed
  };

  // Send bullet data to the server
  socket.send(JSON.stringify({ type: 'SHOOT', ...bullet }));

  // Also add the bullet to the local bullets array for immediate rendering
  bullets.push(bullet);
}
function updateBullets() {
  bullets.forEach(bullet => {
    bullet.x += bullet.speed * Math.cos(bullet.direction);
    bullet.y += bullet.speed * Math.sin(bullet.direction);

    // Remove bullets that go out of bounds (adjust as needed)
    if (bullet.x < 0 || bullet.x > mapSize || bullet.y < 0 || bullet.y > mapSize) {
      bullets = bullets.filter(b => b !== bullet);
    }
  });
}

function renderBullets() {
  bullets.forEach(bullet => {
    gctx.fillStyle = 'red'; // Example color for bullets
    gctx.fillRect(bullet.x - 2, bullet.y - 2, 4, 4); // Draw bullet as a small square
  });
}
let object_options = { 
   x: 0,
   y: 0,
   tex: canvas3,
   wall: false,
   obstacle: false,
   x2: 0,
   y2: 0,
   size: 50,
   solid: false,
   deco: false,
  }

let texture_swap_flag = true

let zoom = 8
let r = 255, g = 255, b = 255
let tinf
let color_picker = false
let erasing = false
let eraseDelay = true
let last_used_colors = []
let drawing_x = 0
let drawing_y = 0
let create_object = false
let picking_texture = false
let button_timeout = false


let hitbox = 0.25
function mapCollision(x, y) {
  x *= speed;
  y *= speed;
  let sizeX = x > 0 ? hitbox : -hitbox
  let sizeY = y > 0 ? hitbox : -hitbox

  if(!texMap.get(map[Math.round(p.x + x + x + sizeX) + mapSize * Math.round(p.y)]).solid && p.x + x > 0 && p.x + x < mapSize -1) 
    p.x += x;
    if(!texMap.get(map[Math.round(p.x) + mapSize * Math.round(p.y + y + sizeY)]).solid && p.y + y > 0 && p.y + y < mapSize -1)
      p.y += y; 
}
ctx.font = "20px arial";

function rankArray(arr) {
  let indexedArray = arr.map((value, index) => ({ value, index }));

  indexedArray.sort((a, b) => b.value - a.value);

  let rankArray = new Array(arr.length);

  indexedArray.forEach((item, rank) => {
    rankArray[item.index] = rank;
  });

  return rankArray;
}

// 0 = mouse in rectangle check
// 1 = alt coordinates in rectangle check
// 2 = mouse in rectangle && mousedown
// 3 = alt coordinates && mousedown
function within(x,y,w,h,state = 0,px,py,){
  switch (state) {
    case 0:
      return mouseX > x && mouseX < x + w && mouseY > y && mouseY < y + h
      case 1:
        return px > x && px < x + w && py > y && py < y + h
        case 2:
          return mouseX > x && mouseX < x + w && mouseY > y && mouseY < y + h && mouseDown
          case 3:
            return px > x && px < x + w && py > y && py < y + h && mouseDown
  }
}

function button(text,x,y,w,h){
  let clicked = false

  gctx.fillStyle = "rgb(10,10,10)"
  gctx.fillRect(x,y,w + 10,h + 10)

  gctx.fillStyle = "rgb(40,40,40)"
  gctx.fillRect(x,y,w,h)

  if(mouseX > x && mouseX < x + w && mouseY > y && mouseY < y + h){
    gctx.fillStyle = "rgb(255,0,0)"
    if(mouseDown) clicked = true
  }
  else gctx.fillStyle = "rgb(255,255,255)"

  gctx.fillText(text,x + 10,y + 15)

  return clicked
}

function getPixelColor(x, y) {
  const imageData = ctx3.getImageData(x, y, 1, 1);
  const data = imageData.data;

  const r = data[0];
  const g = data[1];
  const b = data[2];

  return [r,g,b]
}

// Variables to store mouse position
let mouseX
let mouseY
// Function to get the mouse position relative to the canvas
function getMousePos(canvas, event) {
  const rect = canvas.getBoundingClientRect();
    return {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top
    };
}

// Mouse move event listener
addEventListener('mousemove', (event) => {
    const pos = getMousePos(canvas, event);
    mouseX = pos.x;
    mouseY = pos.y;
});

let mouseDown = false
let buildingBlock = 1


addEventListener('mousedown', (e) => {mouseDown = true;});

addEventListener('mouseup', (e) => {mouseDown = false;});

addEventListener('wheel', (e) => {
   if(draw_shit){ e.deltaY < 0 ? zoom++ : zoom--}

   else {

    if(e.deltaY < 0){
      if(buildingBlock + 1 < texMap.size) buildingBlock++
    }
    else{
      if(buildingBlock > 0) buildingBlock--
    }

   }
  
  
  });

document.addEventListener('mousedown', (e) => {
    if (e.button === 0) { // Left mouse button
      shootBullet();
    }
  });


const key = {
  W: false,
  A: false,
  S: false,
  D: false,
  Q: false,
  E: false,
  " ": false,
};

document.addEventListener("keydown", (e) => {
  const k = e.key.toUpperCase();
  if (key.hasOwnProperty(k)) {
    key[k] = true;
  }
  if (k === "Z") p.r = 0.0001;
});

let draw_shit = false

document.addEventListener("keyup", (e) => {
  const k = e.key.toUpperCase();
  if (key.hasOwnProperty(k)) {
    key[k] = false;
  }
  if (k === "K" && !picking_texture) {
    create_object = !create_object
    draw_shit = false
  }
  if (k === "J" && !picking_texture) {
    draw_shit = !draw_shit
    create_object = false
  }

  if (k === "T") {
    console.log(all_players)
    alert();
  }


  if (k === "X") {
    offCenter = offCenter ? false : true

    if(!offCenter){
      scan1 = scanC1
      scan2 = scanC2
      yCenter = 324
    }
    else{
      scan1 = scanOC1
      scan2 = scanOC2
      yCenter = 450
    }
  }

});

const mapSize = 100 //  prompt("map size?",10)
let map = new Uint8Array(mapSize * mapSize)
const adjacentNWES = [-mapSize,-1,+1, +mapSize]

envi.onload = function() {ctx3.drawImage(envi,0,0)};

setTimeout(() => {
  ctx3.drawImage(envi,0,0)
}, 300);

let texMap = new Map();

texMap.set(0, { x: 48, y: 8, tex: canvas3, wall: false, solid: false, x2: 48, y2: 8 , deco: false});
texMap.set(1, { x: 32, y: 8, tex: canvas3, wall: true, solid: true, x2: 8, y2: 8 , deco: false});

let tex_map_size = 2

let myId

const socket = new WebSocket('127.0.0.1:3000');

// Handle messages from the server
socket.addEventListener('message', (event) => {
      const data = JSON.parse(event.data);

      if(data.type === 'MAP_UPDATE'){
        map[data.location] = data.block
      }
      if(data.type === 'TEXTURE_MAP'){
        texMap = new Map(data.texMap);  // Convert array back to Map

        for (let x = 0; x < texMap.size; x++) {
          let tempObj = texMap.get(x)
          if(tempObj.tex == "canvas3") tempObj.tex = canvas3
          texMap.set(x,tempObj)
        }
      }
      if(data.type === 'NEW_TEXTURE_MAP'){
        
      }

      if (data.type === 'MAP') {
        // Initialize with existing player data
        map = data.map
    }

      if (data.type === 'INIT') {
          // Initialize with existing player data
          initializePlayers(data.players);
      }
      if (data.type === 'UPDATE_PLAYER') {
          // Update player data with the new information
          updatePlayers(data.players);
      }
      if (data.type === 'YOUR_ID') {
        console.log(data)
        myId = data.playerId
        p.tx = ((myId % 10) * 8) % 56
    }
    if (data.type === 'NEW_BULLET') {
      bullets.push(data.bullet);
    } else if (data.type === 'UPDATE_BULLETS') {
      bullets = data.bullets;
    }

});

// Example function to send player data to the server
function sendPlayerData(playerData) {
    socket.send(JSON.stringify({ type: 'UPDATE_PLAYER', playerData }));
}


let playerIdMap = new Map()
let enemyArrayIndex = 0
let all_players

function initializePlayers(players) {

  all_players = players

console.log(players)
  for (const key in players) {
    let numId = Number(key)
    
    if(numId !== myId){
      playerIdMap.set(numId, enemyArrayIndex);
      enemyArrayIndex++

      enemies.push({
        name: players[key].name,   
        tex: char,
        tx: ((numId % 10) * 8) % 56,
        ty: 0,
        x: players[key].x,
        y: players[key].y,
        size: 40,
        spin:0
      },)
  }

  }

}
// EVERY PLAYER CONNECTED TO THE SERVER GETS A UNIQUE ID
// IF THE ID IS NOT IN THE ENEMY ARRAY, IT GETS CREATED

function updatePlayers(players) {

  all_players = players

  for (const key in players) {
    let numId = Number(key)
    let enemyIndex = playerIdMap.get(numId)



  if(numId !== myId){
    if(!playerIdMap.has(numId)){    
      playerIdMap.set(numId, enemyArrayIndex);
        enemyArrayIndex++

        enemies.push({
          name: players[key].name,
          tex: char,
          tx: ((numId % 10) * 8) % 56,
          ty: 0,
          x: players[key].x,
          y: players[key].y,
          size: 40,
          spin:0
        },)
      }
        else{
            enemies[enemyIndex].x = players[key].x
            enemies[enemyIndex].y = players[key].y

            if(enemies[enemyIndex].name == "none"){
              enemies[enemyIndex].name = players[key].name
            }
          }
    }

  }

}

setTimeout(() => {
  setInterval(() => {
    const playerData = {name:namae, x: p.x + 1, y: p.y + 1.5 }
    sendPlayerData(playerData);
  }, 1);
}, 60);

gameLoop();