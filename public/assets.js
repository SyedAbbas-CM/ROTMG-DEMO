// assets.js

export const assets = {};

export function loadAssets() {
  return new Promise((resolve, reject) => {
    const imagesToLoad = [
      { name: 'envi', src: 'assets/images/Oryx/lofi_environment.png' },
      { name: 'char', src: 'assets/images/Oryx/lofi_char.png' },,
      //{ name: 'play', src: 'assets/images/players.png' },
      //{ name: 'env2', src: 'assets/images/lofiEnvironment2.png' },
      { name: 'obj4', src: 'assets/images/Oryx/lofi_obj.png' },
    ];

    let loadedImages = 0;

    imagesToLoad.forEach(imageInfo => {
      const img = new Image();
      img.src = imageInfo.src;
      img.onload = () => {
        assets[imageInfo.name] = img;
        loadedImages++;
        if (loadedImages === imagesToLoad.length) {
          // Load fonts after images
          //loadFonts().then(resolve).catch(reject);
        }
      };
      img.onerror = reject;
    });
  });
}

function loadFonts() {
  return new Promise((resolve, reject) => {
    const agj = new FontFace('agj', 'url(adoquin/adoquin.woff2)');
    agj.load().then(font => {
      document.fonts.add(font);
      resolve();
    }).catch(reject);
  });
}
