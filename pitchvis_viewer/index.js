window.youClicked = async function loadWasm() {
    document.getElementById('instructions').style.display = 'none';
    let say = await import('./pkg')
    say.main_fun();
  }

