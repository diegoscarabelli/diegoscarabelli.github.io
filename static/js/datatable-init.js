/**
 * DataTables initialization for Resources page.
 * Adds sorting and searching functionality to tables.
 */
document.addEventListener('DOMContentLoaded', function() {
    // Initialize DataTables on all tables in the content area
    const tables = document.querySelectorAll('#content table');

    tables.forEach(function(table) {
        // Only initialize if table has a thead (proper table structure)
        if (table.querySelector('thead')) {
            $(table).DataTable({
                // Sorting
                order: [], // No default sorting

                // Search
                searching: true,

                // Pagination - disabled to show all books at once
                paging: false,

                // Info text (e.g., "Showing 1 to 10 of 44 entries")
                info: false,

                // Responsive
                responsive: true,

                // Language customization
                language: {
                    search: "Search books:",
                    searchPlaceholder: "Filter by any column...",
                    zeroRecords: "No matching books found"
                },

                // Column definitions
                columnDefs: [
                    { orderable: true, targets: '_all' }
                ]
            });
        }
    });
});
