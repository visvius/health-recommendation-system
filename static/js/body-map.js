document.addEventListener('DOMContentLoaded', function() {
    // Initialize Select2 in tag mode so that user-added selections persist
    $('#symptomInput').select2({
        placeholder: "Search or select symptoms...",
        allowClear: true,
        dropdownAutoWidth: true,
        minimumInputLength: 0,
        tags: true  // allows custom values to persist even if not in the data list
    });

    // Attach click handlers to all elements with data-body-part attribute
    setTimeout(() => {
        document.querySelectorAll('[data-body-part]').forEach(element => {
            element.addEventListener('click', function() {
                // Visual feedback for the selected body part
                document.querySelectorAll('[data-body-part]').forEach(el => el.classList.remove('selected'));
                this.classList.add('selected');

                const bodyPart = this.getAttribute('data-body-part');
                fetch(`/get_symptoms_by_part?body_part=${encodeURIComponent(bodyPart)}`)
                    .then(response => {
                        console.log(`Fetching symptoms for body part: ${bodyPart}`);
                        return response.json();
                    })
                    .then(fetchedSymptoms => {
                        console.log(`Symptoms for ${bodyPart}:`, fetchedSymptoms);
                        if (!Array.isArray(fetchedSymptoms)) {
                            fetchedSymptoms = [fetchedSymptoms];
                        }

                        const select2 = $('#symptomInput');
                        // Retrieve the previously selected symptoms (tags)
                        const currentSelected = select2.val() || [];
                        
                        // Clear all existing options
                        select2.find('option').remove();
                        
                        // Add new options from the fetched symptoms (visible in the dropdown)
                        fetchedSymptoms.forEach(symptom => {
                            let option = new Option(symptom, symptom, false, false);
                            select2.append(option);
                        });
                        
                        // Add hidden options for any previously selected symptom that is NOT in the fetched list
                        currentSelected.forEach(symptom => {
                            if (!fetchedSymptoms.includes(symptom)) {
                                let option = new Option(symptom, symptom, true, true);
                                // Hide the option so it doesn't show in the dropdown
                                $(option).attr('style', 'display:none;');
                                select2.append(option);
                            }
                        });
                        
                        // Update the select2 with the union of (currentSelected) so that tags remain
                        select2.val(currentSelected).trigger('change');
                        
                        // Open the dropdown for immediate feedback
                        const select2Instance = select2.data('select2');
                        if (select2Instance) {
                            select2Instance.open();
                            setTimeout(() => {
                                select2Instance.dropdown._positionDropdown();
                                select2Instance.dropdown._resizeDropdown();
                            }, 10);
                        }
                        
                        // Focus the search field for usability
                        $('.select2-search__field').trigger('focus');
                    })
                    .catch(error => {
                        console.error(`Error fetching symptoms: ${error}`);
                    });
            });
        });
    }, 100);

    // Close the dropdown when clicking outside
    $(document).on('click', function(e) {
        if (!$(e.target).closest('.select2-container').length) {
            $('#symptomInput').select2('close');
        }
    });
});
