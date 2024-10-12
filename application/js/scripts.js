function SliderController() {
    console.log("slider_controller");
    document.addEventListener('DOMContentLoaded', function () {
        const slider = document.getElementById('custom-slider');
        updateBackgroundSize(slider);
    
        slider.addEventListener('input', function () {
            updateBackgroundSize(slider);
        });
    
        function updateBackgroundSize(slider) {
            const value = (slider.value - slider.min) / (slider.max - slider.min) * 100;
            slider.style.backgroundSize = `${value}% 100%`;
        }
    });
} 


function RadioButtonController(){
    console.log("radio_button_controller");
    document.querySelectorAll('.custom-radio-button input[type="radio"]').forEach(radio => {
        radio.addEventListener('change', function() {
            document.querySelectorAll('.custom-radio-button label').forEach(label => {
                label.style.backgroundColor = 'transparent';
            });
            if (this.checked) {
                console.log("Checked")
                this.closest('label').style.backgroundColor = '#6c18ff';
            }
        });
    });
}

SliderController()
RadioButtonController()