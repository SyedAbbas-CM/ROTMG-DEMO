//import map from "./mapd.js";
const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d", { alpha: false });
ctx.imageSmoothingEnabled = false;
ctx.webkitImageSmoothingEnabled = false;

const gameCanvas = new OffscreenCanvas(600,600)
const gctx = gameCanvas.getContext("2d", { alpha: false });
gctx.imageSmoothingEnabled = false;
gctx.webkitImageSmoothingEnabled = false;

const envi = new Image();
envi.src = "sprites/lofiEnvironment.png";
const char = new Image();
char.src = "sprites/lofiChar.png";
const play = new Image();
play.src = "sprites/players.png";
const env2 = new Image();
env2.src = "sprites/lofiEnvironment2.png";
const obj2 = new Image();
obj2.src = "sprites/lofiObj2.png";

const agj = new FontFace("agj", "url(adoquin/adoquin.woff2)");
agj.load().then((font) => {
  document.fonts.add(font);
  gctx.font = "25px agj"
});

const p = { name: "surfin" , x: 50, y: 50, r: 0.003, tx: 0, ty: 0 }; // r:0 causes walls to not be drawn

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

let game_state = "game"

const stateActions = {
  game: game,
};

function gameLoop() {

  const action = stateActions[game_state]
  
  action()

 // map[Math.round(p.x) + mapSize * (Math.round(p.y))] = 2

  ctx.drawImage(gameCanvas,0,0)
  requestAnimationFrame(gameLoop);
}
let projectile_cooldown = -100
let projectile_timeout = 10
function game(){
  projectile_cooldown++
  if(mouseDown && projectile_cooldown > projectile_timeout){
    projectile_cooldown = 0
    let tan = Math.atan2(300 - mouseY, 300  - mouseX) - p.r
    socket.send(JSON.stringify({ type: 'NEW_PROJECTILE', tan }));
  }

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
  const time_since = Date.now() - update_timestamp
  const how_long_to_move = time_since / 100

  for (const key in old_player_list) {
    if (Number(key) !== MY_ID && Object.prototype.hasOwnProperty.call(player_list, key)) {
      const e = old_player_list[key];
      const e2 = player_list[key]

      const dx = e.x - e2.x
      const dy = e.y - e2.y

      const approx_x = e.x - (dx * how_long_to_move)
      const approx_y = e.y - (dy * how_long_to_move)

      if(approx_x > nearPX1 &&
        approx_x < nearPX2 &&
        approx_y > nearPY1 &&
        approx_y < nearPY2
      ){
  
        let offCenterPlus = offCenter ? offCenterDiff : 0
        let screenX = (tSize * (approx_x - p.x + 5)) - xCenter
        let screenY = (tSize * (approx_y - p.y + 5)) - yCenter + offCenterPlus
  
        let rotatedScreenX = screenX * cos + screenY * -sin + xCenter
        let rotatedScreenY = screenX * sin + screenY * cos + yCenter
  
        entity.push({
            not_me: true,
            name: p.name,
            x: rotatedScreenX,
            y: rotatedScreenY,
            tex: char,
            tx: 0,
            ty: 0,
            size: 40,
        })

      }

    }
  }

  projectile_list = projectile_list.filter(e => e.lifetime > 0);

  projectile_list.forEach(e =>{
    e.x -= Math.cos(e.tan) * e.spd
    e.y -= Math.sin(e.tan) * e.spd
    e.lifetime--

    map[Math.floor(e.x-1) + mapSize * (Math.floor(e.y-1))] = 2
    setTimeout(() => {
      map[Math.floor(e.x -1) + mapSize * (Math.floor(e.y-1))] = 0
    }, 100);

    let offCenterPlus = offCenter ? offCenterDiff : 0
    let screenX = (tSize * (e.x - p.x + 5 )) - xCenter
    let screenY = (tSize * (e.y - p.y + 5)) - yCenter + offCenterPlus

    let rotatedScreenX = screenX * cos + screenY * -sin + xCenter - 20
    let rotatedScreenY = screenX * sin + screenY * cos + yCenter - 35

    entity.push({
        projectile: true,
        tan: e.tan,
        x: rotatedScreenX,
        y: rotatedScreenY,
        tex: obj2,
        tx: 56,
        ty: 48,
        size: 32,
    })
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
  
            // Calculate the edges before rotation
            let edgeX1 = tileX + wallEdge[n4 + 0];
            let edgeY1 = tileY + wallEdge[n4 + 1];
            let edgeX2 = tileX + wallEdge[n4 + 2];
            let edgeY2 = tileY + wallEdge[n4 + 3];
            
            // Apply rotation
            let rotatedEdgeX1 = edgeX1 * cos + edgeY1 * -sin;
            let rotatedEdgeY1 = edgeX1 * sin + edgeY1 * cos;
            let rotatedEdgeX2 = edgeX2 * cos + edgeY2 * -sin;
            let rotatedEdgeY2 = edgeX2 * sin + edgeY2 * cos;
            
            // Update the arrays
            wallStartX[n] = Math.round(rotatedEdgeX1 + xCenter);
            wallStartY[n] = rotatedEdgeY1 + yCenter;
            wallEndX[n] = rotatedEdgeX2 + xCenter;
            wallEndY[n] = rotatedEdgeY2 + yCenter;
          
            wallWidth[n] = Math.abs(rotatedEdgeX1 - rotatedEdgeX2);
            lowestPoint[n] = Math.max(rotatedEdgeY1, rotatedEdgeY2);
            wallRatio[n] = (rotatedEdgeY1 - rotatedEdgeY2) / (rotatedEdgeX1 - rotatedEdgeX2);

          }

          let rank = rankArray(lowestPoint); //no walls on 0 degree problem

            for (let g = 0; g < 4; g++) { 
              if (rank[g] < 2 && texMap.has(map[arrayLocation + adjacentNWES[g]]) && !texMap.get(map[arrayLocation + adjacentNWES[g]]).wall){
                // if undefined adjacant = carsh

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
        if(e.projectile){
          gctx.save();
          gctx.translate( e.x + e.size / 2 ,  e.y + e.size / 2);
          gctx.rotate(e.tan - 2.35619 + p.r);
          gctx.drawImage(e.tex,e.tx,e.ty,8,8, -e.size / 2, -e.size / 2, e.size, e.size);
          gctx.restore();
        }
        else {
        gctx.drawImage(e.tex,e.tx,e.ty,8,8,Math.round(e.x) - e.size * 0.5,Math.round(e.y) - e.size,e.size,e.size);
        if(e.not_me){
        gctx.fillStyle = "rgb(255,255,255)"
        gctx.fillText(e.name,e.x - 20,e.y + 20)
      }}
      }
  });

  if(roofs.length){
    gctx.save();
    gctx.translate(xCenter, yCenter - tSize );
    gctx.rotate(p.r);
    roofs.forEach( e =>{gctx.drawImage(e.tex, e.tx, e.ty, 8, 8, e.x - 1, e.y - 1, tSize+2, tSize+2);})
    gctx.restore();
  }
} // END OF GAME

let object_options = { 
   x: 0,
   y: 0,
   tex: envi,
   wall: false,
   obstacle: false,
   x2: 0,
   y2: 0,
   size: 50,
   solid: false,
   deco: false,
  }


let button_timeout = true

let hitbox = 0.25
function mapCollision(x, y) {
  x *= speed;
  y *= speed;

  let sizeX = x > 0 ? hitbox : -hitbox
  let sizeY = y > 0 ? hitbox : -hitbox
  
  const array_location_x = Math.round(p.x + x + sizeX) + mapSize * Math.round(p.y)
  const array_location_y = Math.round(p.x) + mapSize * Math.round(p.y + y + sizeY)

  if(!texMap.get(map[array_location_x]).solid) 
    p.x += x;

  if(!texMap.get(map[array_location_y]).solid)
    p.y += y; 
}

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

const BUTTON_HOVER_COLOR = "rgb(255,0,0)";
const BUTTON_DEFAULT_COLOR = "rgb(40,40,40)";
const BUTTON_BORDER_COLOR = "rgb(10,10,10)";
const BUTTON_TEXT_COLOR = "rgb(255,255,255)";

function button(text,x,y,w,h){
  let clicked = false

    gctx.fillStyle = BUTTON_BORDER_COLOR
    gctx.fillRect(x,y,w + 10,h + 10)

    gctx.fillStyle = BUTTON_DEFAULT_COLOR
    gctx.fillRect(x,y,w,h)

    if(mouseX > x && mouseX < x + w && mouseY > y && mouseY < y + h){
      gctx.fillStyle = BUTTON_HOVER_COLOR
      if(mouseDown) {
        clicked = true
        button_timeout = false
        setTimeout(() => {button_timeout = true}, 500);
      }
    }
    else gctx.fillStyle = BUTTON_TEXT_COLOR

    gctx.fillText(text,x + 10,y + 15)

  return clicked
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

const key = {
  W: false,
  A: false,
  S: false,
  D: false,
  Q: false,
  E: false,
  " ": false,
};

window.addEventListener("keydown", (e) => {
  const k = e.key.toUpperCase();
  if (key.hasOwnProperty(k)) {
    key[k] = true;
  }
});


window.addEventListener("keyup", (e) => {
  const k = e.key.toUpperCase();

  // Update key state
  if (key.hasOwnProperty(k)) {
    key[k] = false;
  }

  // Handle game state transitions
  switch (k) {
    case "T":
      console.log(player_list)
      alert("pause.");
      break;
    case "Z":
      p.r = 0.0001;
      break;

    case "X":
      offCenter = !offCenter;
      
      // Toggle scan and yCenter values based on offCenter
      if (offCenter) {
        scan1 = scanOC1;
        scan2 = scanOC2;
        yCenter = 450;
      } else {
        scan1 = scanC1;
        scan2 = scanC2;
        yCenter = 324;
      }
      break;
  }
});


const mapSize = 100 //  prompt("map size?",10)
let map = new Uint8Array(mapSize * mapSize)
const adjacentNWES = [-mapSize,-1,+1, +mapSize]


map[52 + (52 * 100)] = 1

let texMap = new Map();

texMap.set(0, { x: 48, y: 8, tex: envi, wall: false, solid: false, x2: 48, y2: 8 , deco: false});
texMap.set(1, { x: 32, y: 8, tex: envi, wall: true, solid: true, x2: 8, y2: 8 , deco: false});
texMap.set(2, { x: 32, y: 8, tex: envi, wall: false, solid: false, x2: 48, y2: 8 , deco: false});

let MY_ID, player_list, old_player_list, update_timestamp, projectile_list = []
const socket = new WebSocket('ws://localhost:3000');

socket.addEventListener('message', (event) => {
    let data = event.data

    try {
        data = JSON.parse(event.data);  // Attempt to parse the data
    } catch (error) {
        console.error("Malformed JSON received:", error);
        return;  // Stop further processing
    }

      switch (data.type) {
        case 'INIT':
          MY_ID = data.playerId
          player_list = data.players // an object containing objects labeled with the Ids; players = { 1:{x:82,y:52}, 2:{x:42,y:14} }
          console.log(player_list)
          break;

        case 'UPDATE_PLAYER':
          update_timestamp = Date.now()

          old_player_list = player_list
          player_list = data.players
          break;

        case 'NEW_PROJECTILE':
        projectile_list.push(data.projectile)
          break;
      }
});

function sendPlayerData(playerData) {
    socket.send(JSON.stringify({ type: 'UPDATE_PLAYER', playerData }));
}

setTimeout(() => {
  setInterval(() => {
    const playerData = {name: p.name, x: p.x + 1, y: p.y + 1.5 }
    sendPlayerData(playerData);
  }, 100);
}, 1000);

gameLoop();
