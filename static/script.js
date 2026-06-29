const imageInput =
document.getElementById("imageInput");

const preview =
document.getElementById("preview");

const output =
document.getElementById("output");


// IMAGE PREVIEW

imageInput.addEventListener(
    "change",
    function () {

        const file =
            imageInput.files[0];

        if (file) {

            preview.style.display = "block";

            preview.src =
                URL.createObjectURL(file);
        }
    }
);


// CLEAR BUTTON

document
.querySelector(".clear-btn")
.addEventListener(
    "click",
    function () {

        output.value = "";

        preview.src = "";

        preview.style.display = "none";

        imageInput.value = "";
    }
);


// TRANSCRIBE BUTTON

document
.querySelector(".transcribe-btn")
.addEventListener(
    "click",
    async function () {

        const file =
            imageInput.files[0];

        if (!file) {

            alert(
                "Please select an image first."
            );

            return;
        }

        output.value =
            "Reading handwriting...";

        const formData =
            new FormData();

        formData.append(
            "image",
            file
        );

        const response =
            await fetch(
                "/recognize",
                {
                    method: "POST",
                    body: formData
                }
            );

        const data =
            await response.json();

        output.value =
            data.text;
    }
);