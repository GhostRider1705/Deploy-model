submitButton = document.querySelector('#submit');



const circlechart = document.querySelectorAll('.circlechart');





submitButton.addEventListener('submit', (e) => {
    // e.preventDefault();
    circlechart.forEach(item => item.style.display = 'none');
});